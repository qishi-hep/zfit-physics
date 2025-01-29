#!/bin/bash
# script has to be executed inside folder `docs`
# get current directory name
pushd `dirname $0` > /dev/null
MAKE_DOCS_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
popd > /dev/null

# generate the ReST files
#echo ${MAKE_DOCS_PATH}/../zfit_physics
ls ${MAKE_DOCS_PATH}
ls .
sphinx-apidoc -o ${MAKE_DOCS_PATH}/api ${MAKE_DOCS_PATH}/../src/zfit_physics -fMeT && \
python3 ${MAKE_DOCS_PATH}/api/tools/change_headline.py ${MAKE_DOCS_PATH}/api/zfit_physics.* && \
make -C ${MAKE_DOCS_PATH} clean && make -C ${MAKE_DOCS_PATH} html -j8 && \
echo "Documentation successfully built!" || echo "FAILED to build Documentation"
