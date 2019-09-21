from pip._vendor import pkg_resources
import sys

def get_version(package):
    package = package.lower()
    return next((p.version for p in pkg_resources.working_set if p.project_name.lower() == package), "No match")

package=sys.argv[1]
print(get_version(package))
