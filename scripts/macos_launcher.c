// Locate bundled runtime resources without depending on the install location.
#include <mach-o/dyld.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    char executable[PATH_MAX], directory[PATH_MAX], path[PATH_MAX * 2];
    uint32_t size = sizeof(executable);
    if (_NSGetExecutablePath(executable, &size) || !realpath(executable, directory)) return 1;
    char *slash = strrchr(directory, '/');
    if (!slash) return 1;
    *slash = 0;
    snprintf(path, sizeof(path), "%s/../Resources/vulkan/icd.d/MoltenVK_icd.json", directory);
    setenv("VK_ICD_FILENAMES", path, 1);
    setenv("VK_DRIVER_FILES", path, 1);
    snprintf(path, sizeof(path), "%s:%s", directory, getenv("PATH") ? getenv("PATH") : "/usr/bin:/bin");
    setenv("PATH", path, 1);
    snprintf(path, sizeof(path), "%s/PhotaraStudio", directory);
    argv[0] = path;
    execv(path, argv);
    perror("Cannot launch Photara Studio");
    return 1;
}
