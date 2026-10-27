#!/usr/bin/env python3
"""
Strict dependency resolvers for the AKG v2 rebuild (shared library).

Design principles (all systems):
  * The resolution index is every file on disk under the source tree with a
    configured extension (os.walk pruning ignore_dirs / dot-dirs, like
    pipeline/generate_all.py:generate_files, but over the whole tree instead of
    module dirs only).  An optional set of ignore names can be "un-ignored"
    (linux: 'include').
  * A link is produced only when the statement itself identifies the target
    (path / FQN / dotted module), or a bare name is unique.  Anything else is
    dropped with a reason.  No longest-common-prefix fallback, no simple-name
    or last-segment fallback.
  * Module mapping is done by the caller (file -> module dict).

Every resolve() returns (target_or_None, reason).
"""

import os
import posixpath
import sys
from collections import defaultdict, Counter

# --------------------------------------------------------------------------
# Disk index
# --------------------------------------------------------------------------


def build_disk_index(repo, extensions, ignore_dirs, unignore=()):
    ignore = set(ignore_dirs or ()) - set(unignore)
    exts = tuple(extensions) if extensions else None
    out = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in ignore and not d.startswith(".")]
        for fn in files:
            if exts and not fn.endswith(exts):
                continue
            out.append(os.path.relpath(os.path.join(root, fn), repo).replace("\\", "/"))
    return out


class FileIndex:
    def __init__(self, paths):
        self.paths = set(paths)
        self.by_base = defaultdict(list)
        for p in self.paths:
            self.by_base[posixpath.basename(p)].append(p)
        self._suffix = None

    def __contains__(self, p):
        return p in self.paths

    def build_suffix(self, wanted=None, max_seg=12):
        """path suffix -> [paths].  If `wanted` given, only those keys are stored."""
        if wanted is not None:
            wanted = set(wanted)
        suf = defaultdict(list)
        for p in self.paths:
            parts = p.split("/")
            for k in range(1, min(len(parts), max_seg) + 1):
                s = "/".join(parts[-k:])
                if wanted is None or s in wanted:
                    suf[s].append(p)
        self._suffix = suf
        return suf

    def suffix(self, s):
        return self._suffix.get(s, []) if self._suffix is not None else \
            [p for p in self.paths if p == s or p.endswith("/" + s)]


# --------------------------------------------------------------------------
# C / C++
# --------------------------------------------------------------------------

CPP_STD_HEADERS = set("""
algorithm any array atomic barrier bit bitset cassert ccomplex cctype cerrno cfenv cfloat
charconv chrono cinttypes ciso646 climits clocale cmath codecvt compare complex concepts
condition_variable coroutine csetjmp csignal cstdalign cstdarg cstdbool cstddef cstdint cstdio
cstdlib cstring ctgmath ctime cuchar cwchar cwctype deque exception execution expected
filesystem format forward_list fstream functional future initializer_list iomanip ios iosfwd
iostream istream iterator latch limits list locale map memory memory_resource mutex new
numbers numeric optional ostream print queue random ranges ratio regex scoped_allocator
semaphore set shared_mutex source_location span spanstream sstream stack stacktrace
stdexcept stdfloat stop_token streambuf string string_view strstream syncstream
system_error thread tuple type_traits typeindex typeinfo unordered_map unordered_set
utility valarray variant vector version
assert.h complex.h ctype.h errno.h fenv.h float.h inttypes.h iso646.h limits.h locale.h
math.h setjmp.h signal.h stdalign.h stdarg.h stdatomic.h stdbool.h stddef.h stdint.h
stdio.h stdlib.h stdnoreturn.h string.h tgmath.h threads.h time.h uchar.h wchar.h wctype.h
aio.h arpa cpio.h dirent.h dlfcn.h fcntl.h fmtmsg.h fnmatch.h ftw.h glob.h grp.h iconv.h
langinfo.h libgen.h monetary.h mqueue.h ndbm.h netdb.h nl_types.h poll.h pthread.h pwd.h
regex.h sched.h search.h semaphore.h spawn.h strings.h stropts.h syslog.h tar.h termios.h
trace.h ulimit.h unistd.h utime.h utmpx.h wordexp.h malloc.h memory.h alloca.h getopt.h
execinfo.h err.h endian.h byteswap.h features.h values.h io.h process.h direct.h windows.h
winsock2.h ws2tcpip.h conio.h intrin.h immintrin.h xmmintrin.h emmintrin.h pmmintrin.h
smmintrin.h tmmintrin.h nmmintrin.h x86intrin.h arm_neon.h omp.h mpi.h
""".split())


def _norm_rel(srcdir, inc):
    p = posixpath.normpath(posixpath.join(srcdir, inc)) if srcdir else posixpath.normpath(inc)
    if p.startswith("../") or p == ".." or p.startswith("/"):
        return None
    return p


class CppResolver:
    """Generic strict C/C++ include resolver.

    path include (contains '/'):
        candidates = index files equal to / ending with '/'+inc, plus the path
        relative to the source dir, plus extra include roots (hook).
        1 candidate -> take.  Several: relative-to-source wins; else unique
        candidate sharing the source's top-level component; else drop.
    bare include:
        same dir; standard header -> drop; unique basename; else drop.
    """

    def __init__(self, index, std_headers=CPP_STD_HEADERS):
        self.idx = index
        self.std = std_headers

    def prepare(self, includes):
        self.idx.build_suffix({i for i in includes if "/" in i and ".." not in i.split("/")})

    def extra_roots(self, src):
        return ()

    def special(self, src, inc):
        """hook: return (target, reason) to decide, or None to continue."""
        return None

    def resolve(self, src, inc):
        inc = inc.strip()
        if not inc:
            return None, "empty"
        srcdir = posixpath.dirname(src)
        sp = self.special(src, inc)
        if sp is not None:
            return sp
        if "/" in inc:
            rel = _norm_rel(srcdir, inc)
            cands = set()
            if ".." not in inc.split("/") and not inc.startswith("./"):
                cands.update(self.idx.suffix(inc))
            if rel and rel in self.idx:
                cands.add(rel)
            for r in self.extra_roots(src):
                p = posixpath.normpath(r + "/" + inc)
                if p in self.idx:
                    cands.add(p)
            if not cands:
                return None, "path_not_found"
            if len(cands) == 1:
                return next(iter(cands)), "path_unique"
            if rel in cands:
                return rel, "path_relative_preferred"
            top = src.split("/")[0]
            same_top = [c for c in cands if c.split("/")[0] == top]
            if len(same_top) == 1:
                return same_top[0], "path_same_top_preferred"
            return None, "path_ambiguous"
        same = posixpath.join(srcdir, inc) if srcdir else inc
        if same in self.idx:
            return same, "bare_same_dir"
        if inc in self.std:
            return None, "std_header"
        cands = self.idx.by_base.get(inc, [])
        if len(cands) == 1:
            return cands[0], "bare_unique_basename"
        if not cands:
            return None, "bare_not_found"
        return None, "bare_ambiguous"


class LinuxCppResolver(CppResolver):
    """Linux kernel include semantics (Kbuild LINUXINCLUDE order):
         arch/X/include, arch/X/include/generated, include,
         arch/X/include/uapi, arch/X/include/generated/uapi, include/uapi
       '<asm/...>': arch/X/include{,/generated,/uapi,/generated/uapi}/asm/... when
       the source is under arch/X; otherwise (or if absent: Kbuild generic-y)
       include/asm-generic/<rest>; else drop.
       A path include whose first component is a kernel include namespace
       (any subdir of include/, include/uapi/, arch/*/include[/uapi]) that is not
       found in the roots is dropped (no suffix fallback); other path includes
       use the generic rule.
    """

    def __init__(self, index, std_headers=CPP_STD_HEADERS):
        super().__init__(index, std_headers)
        ns = set()
        for p in index.paths:
            parts = p.split("/")
            if parts[0] == "include" and len(parts) > 2:
                ns.add(parts[1])
                if parts[1] == "uapi" and len(parts) > 3:
                    ns.add(parts[2])
            elif parts[0] == "arch" and len(parts) > 4 and parts[2] == "include":
                ns.add(parts[3])
                if parts[3] in ("uapi", "generated") and len(parts) > 5:
                    ns.add(parts[4])
        ns.discard("uapi")
        ns.discard("generated")
        self.namespaces = ns
        ov = set()
        for p in self.idx.paths:
            parts = p.split("/")
            if parts[0] == "arch" and len(parts) > 4 and parts[2] == "include":
                q = parts[3:]
                while q and q[0] in ("generated", "uapi"):
                    q = q[1:]
                if len(q) > 1 and q[0] == "asm":
                    ov.add("/".join(q[1:]))
        self.arch_asm_overrides = ov

    @staticmethod
    def _arch(src):
        parts = src.split("/")
        return parts[1] if parts[0] == "arch" and len(parts) > 2 else None

    def _roots(self, arch):
        r = []
        if arch:
            r += [f"arch/{arch}/include", f"arch/{arch}/include/generated"]
        r.append("include")
        if arch:
            r += [f"arch/{arch}/include/uapi", f"arch/{arch}/include/generated/uapi"]
        r.append("include/uapi")
        return r

    def special(self, src, inc):
        if "/" not in inc:
            return None
        arch = self._arch(src)
        srcdir = posixpath.dirname(src)
        if inc.startswith("asm/"):
            rest = inc[4:]
            if arch:
                for r in (f"arch/{arch}/include", f"arch/{arch}/include/generated",
                          f"arch/{arch}/include/uapi", f"arch/{arch}/include/generated/uapi"):
                    if f"{r}/asm/{rest}" in self.idx:
                        return f"{r}/asm/{rest}", "linux_asm_arch"
            elif rest in self.arch_asm_overrides:
                return None, "linux_asm_nonarch_arch_overridden"
            g = f"include/asm-generic/{rest}"
            if g in self.idx:
                return g, ("linux_asm_generic_fallback" if arch else "linux_asm_generic_nonarch")
            g = f"include/uapi/asm-generic/{rest}"
            if g in self.idx:
                return g, ("linux_asm_generic_fallback" if arch else "linux_asm_generic_nonarch")
            return None, ("linux_asm_arch_not_found" if arch else "linux_asm_nonarch_dropped")
        rel = _norm_rel(srcdir, inc)
        if rel and rel in self.idx:
            return rel, "linux_relative"
        for r in self._roots(arch):
            p = f"{r}/{inc}"
            if p in self.idx:
                return p, "linux_include_root"
        if inc.split("/")[0] in self.namespaces:
            return None, "linux_root_not_found"
        return None   # generic rule


# --------------------------------------------------------------------------
# Java
# --------------------------------------------------------------------------


class JavaResolver:
    """FQN a.b.C -> unique index path ending with a/b/C.java.
    Nested / static-member imports: trailing segments are stripped while the
    remaining last segment is a class-like (Capitalised) name.
    Wildcard imports never reach here (parse_java_imports does not capture '*')."""

    def __init__(self, index):
        self.idx = index
        self.idx.build_suffix()

    def resolve(self, src, fqn):
        parts = [p for p in fqn.strip().split(".") if p]
        if len(parts) < 2:
            return None, "java_too_short"
        for k in range(len(parts), 1, -1):
            if k < len(parts) and not parts[k - 1][:1].isupper():
                break
            key = "/".join(parts[:k]) + ".java"
            c = self.idx.suffix(key)
            if c:
                if len(c) == 1:
                    return c[0], ("java_fqn" if k == len(parts) else "java_fqn_outer_class")
                return None, "java_ambiguous"
        return None, "java_not_found"


# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------


STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"test", "tests"}


class PythonResolver:
    """Package roots are detected from __init__.py files: a directory D with
    __init__.py whose parent has none is a top-level package named basename(D)
    living under root dirname(D).

    absolute 'x.y.z': x must be a project top-level package (else drop: stdlib /
      third-party).  Try <root>/x/y/z.py and <root>/x/y/z/__init__.py for every
      root of x.  One hit -> take; several -> the root that is an ancestor of the
      source, if unique; else drop.  No parent-symbol / last-segment fallback.
    script-dir import: if the source dir is not a package (no __init__.py) and
      contains x.py or x/ package, sys.path[0] semantics resolve it there.
    relative '.a.b' / '..a': resolved against the source's package.
    """

    def __init__(self, index):
        self.idx = index
        pkgdirs = {posixpath.dirname(p) for p in index.paths if posixpath.basename(p) == "__init__.py"}
        self.pkgdirs = pkgdirs
        self.toplevel = defaultdict(list)
        for d in pkgdirs:
            parent = posixpath.dirname(d)
            if parent not in pkgdirs:
                self.toplevel[posixpath.basename(d)].append(parent)

    def _forms(self, base):
        return (base + ".py", base + "/__init__.py")

    def _hit(self, base):
        for f in self._forms(base):
            if f in self.idx:
                return f
        return None

    def resolve(self, src, mod):
        mod = mod.strip()
        srcdir = posixpath.dirname(src)
        if mod.startswith("."):
            lvl = len(mod) - len(mod.lstrip("."))
            pkg = srcdir.split("/") if srcdir else []
            if lvl - 1 > len(pkg):
                return None, "py_relative_out_of_tree"
            base = pkg[:len(pkg) - (lvl - 1)]
            rest = [r for r in mod.lstrip(".").split(".") if r]
            path = "/".join(base + rest)
            if not rest:
                t = path + "/__init__.py" if path else "__init__.py"
                return (t, "py_relative") if t in self.idx else (None, "py_relative_not_found")
            h = self._hit(path)
            return (h, "py_relative") if h else (None, "py_relative_not_found")
        parts = [p for p in mod.split(".") if p]
        if not parts:
            return None, "empty"
        top = parts[0]
        tail = "/".join(parts)
        # script directory (sys.path[0]) for non-package scripts
        if (srcdir + "/__init__.py" if srcdir else "__init__.py") not in self.idx:
            sd = (srcdir + "/" if srcdir else "")
            if self._hit(sd + top):
                h = self._hit(sd + tail)
                return (h, "py_script_dir") if h else (None, "py_not_found")
        roots = self.toplevel.get(top)
        if not roots:
            return None, "py_external"
        # repo-root packages are always importable; a nested root only when it is an
        # ancestor of the source and the name does not shadow a stdlib module
        roots = [r for r in roots if r == "" or
                 (src.startswith(r + "/") and top not in STDLIB)]
        if not roots:
            return None, "py_external_nested_root"
        hits = {}
        for r in roots:
            h = self._hit((r + "/" if r else "") + tail)
            if h:
                hits[r] = h
        if not hits:
            return None, "py_not_found"
        if len(hits) == 1:
            return next(iter(hits.values())), "py_module"
        anc = [h for r, h in hits.items() if r == "" or src.startswith(r + "/")]
        if len(anc) == 1:
            return anc[0], "py_module_ancestor_root"
        return None, "py_ambiguous_root"


# --------------------------------------------------------------------------
# Aggregation / relabeling helpers
# --------------------------------------------------------------------------


def module_edges(links, f2m):
    c = Counter()
    for s, t in links:
        a, b = f2m.get(s), f2m.get(t)
        if a and b and a != b:
            c[(a, b)] += 1
    return c


def relabel_dp(items, edges):
    out = []
    for it in items:
        n = dict(it)
        n["label"] = int((it["module_A"], it["module_B"]) in edges)
        out.append(n)
    return out


def relabel_ar(items, edges):
    """Items kept verbatim; added 'label' (task edge still in the new graph) and
    'new_target_set' (full outgoing set of source_module in the new graph)."""
    outg = defaultdict(set)
    for a, b in edges:
        outg[a].add(b)
    out = []
    for it in items:
        n = dict(it)
        n["label"] = int((it["source_module"], it["target_module"]) in edges)
        n["new_target_set"] = sorted(outg.get(it["source_module"], ()))
        out.append(n)
    return out


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 1.0
