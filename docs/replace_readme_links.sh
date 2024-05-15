#!/bin/bash

###
#
#   Replaces the links appearing in README.md with Doxygen @ref links.
#   The point of this is to be able to reuse the README.md as the front page of the Doxygen
#   documentation, while keeping the usefulness of the README.md file as the landing page
#   on Github.
#
###

WEBSITE='https://warpii.com'

# Use sed in-place editing, with the .bak "backup" extension.
sed -i '.bak' "s|${WEBSITE}/install.html|@ref install|g" README.md 
sed -i '.bak' "s|${WEBSITE}/warpxm_run.html|@ref warpxm_run|g" README.md 
sed -i '.bak' "s|${WEBSITE}/warpxm_user_guide.html|@ref warpxm_user_guide|g" README.md 
sed -i '.bak' "s|${WEBSITE}/warpxm_cmake.html|@ref warpxm_cmake|g" README.md 
sed -i '.bak' "s|${WEBSITE}/warpxm_structure.html|@ref warpxm_structure|g" README.md 
sed -i '.bak' "s|${WEBSITE}/warpxm_apps.html|@ref warpxm_apps|g" README.md 
sed -i '.bak' "s|${WEBSITE}/warpy_dev.html|@ref warpy_dev|g" README.md 
sed -i '.bak' "s|${WEBSITE}/coding_standards.html|@ref coding_standards|g" README.md 
sed -i '.bak' "s|${WEBSITE}/commenting.html|@ref commenting|g" README.md 
