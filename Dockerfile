FROM openwhisk/action-python-v3.9

RUN pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

