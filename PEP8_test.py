import pycodestyle as pep8
import os

cwd = os.getcwd()

print('\n')
print('##############################')
print('######### PEP8 Checks ########')
print('##############################')
print('\n')

for path, subdirs, files in os.walk(cwd):
    for name in files:
        if name.endswith('.py'):
            print('----------------', name)
            fchecker = pep8.Checker(path+'\\'+name, show_source=True)
            file_errors = fchecker.check_all()
            if file_errors > 0:
                print("Found %s errors (and warnings)" % file_errors)
