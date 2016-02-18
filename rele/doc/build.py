#!/usr/bin/env python
# Build the documentation.

import sys, os
from subprocess import check_call, check_output, CalledProcessError, Popen, PIPE

def build_docs(srcPath):
  print(srcPath)
  print(os.path.join(srcPath, 'include/rele/'))
  # Build docs.
  # cmd = ['doxygen', '-']
  # p = Popen(cmd, stdin=PIPE)
  # p.communicate(input=r'''
  #     PROJECT_NAME      = ReLe
  #     GENERATE_LATEX    = NO
  #     GENERATE_MAN      = NO
  #     GENERATE_RTF      = NO
  #     CASE_SENSE_NAMES  = NO
  #     INPUT             = {0}
  #     FILE_PATTERNS     = *.h
  #     RECURSIVE         = YES
  #     QUIET             = YES
  #     JAVADOC_AUTOBRIEF = NO
  #     AUTOLINK_SUPPORT  = NO
  #     GENERATE_HTML     = NO
  #     GENERATE_XML      = YES
  #     XML_OUTPUT        = {1}/doxyxml
  #     HTML_OUTPUT       = {1}/doxyhtml
  #     ALIASES           = "rst=\verbatim embed:rst"
  #     ALIASES          += "endrst=\endverbatim"
  #     MACRO_EXPANSION   = YES
  #     PREDEFINED        = _WIN32=1 \
  #                         FMT_USE_VARIADIC_TEMPLATES=1 \
  #                         FMT_USE_RVALUE_REFERENCES=1 \
  #                         FMT_USE_USER_DEFINED_LITERALS=1 \
  #                         FMT_API=
  #     EXCLUDE_SYMBOLS   = fmt::internal::* StringValue write_str
  #   '''.format(os.path.join(srcPath, 'include/rele/'),os.path.join(srcPath, 'doc/build')).encode('UTF-8'))
  b = Popen(['make', 'html'], stdin=PIPE, cwd=os.path.join(srcPath, 'doc/'))
  b.communicate(input=r' ')

if __name__ == '__main__':
  build_docs(sys.argv[1])
