#include "utilities.h"

#include <deal.II/base/numbers.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <limits.h>
#include <unistd.h>
#include <sys/stat.h>

void show_for_debug(bool val)
{
    std::cout << val << std::endl;
}

void show_for_debug(int val)
{
    std::cout << val << std::endl;
}

void show_for_debug(unsigned int val)
{
    std::cout << val << std::endl;
}

void show_for_debug(double val)
{
    std::cout << std::setprecision(15) << val << std::endl;
}

void show_for_debug(size_t val)
{
    std::cout << val << std::endl;
}

void show_for_debug(const std::vector<double>& val)
{
    std::cout << "[";
    if (val.size() >= 1)
    {
        std::cout << val[0];
        for (size_t i = 1; i < val.size(); i++)
        {
            std::cout << ", " << val[i];
        }
    }
    std::cout << "]" << std::endl;
}

void show_for_debug(const dealii::VectorizedArray<double>& val)
{
    std::cout << "[";
    if (val.size() >= 1)
    {
        std::cout << val[0];
        for (size_t i = 1; i < val.size(); i++)
        {
            std::cout << ", " << val[i];
        }
    }
    std::cout << "]" << std::endl;
}

// Thank you ChatGPT
std::string remove_file_extension(std::string filename) {
    // Find the last occurrence of a slash in the filename
    size_t last_slash = filename.find_last_of('/');
    std::string file_part;

    if (last_slash == std::string::npos) {
        file_part = filename;
    } else {
        file_part = filename.substr(last_slash+1, filename.size());
    }

    // Find the last occurrence of a period in the filename
    size_t last_dot = file_part.find_last_of('.');
    
    // If no period is found, return the original filename
    if (last_dot == std::string::npos) {
        return file_part;
    }

    // Return the substring from the start to the character before the last period
    return file_part.substr(0, last_dot);
}

void create_and_move_to_subdir(const std::string subdir) {
    // Get and print the current working directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
    } else {
        std::cerr << "getcwd() error: " << strerror(errno) << std::endl;
        exit(1);
    }
    std::stringstream new_dir_stream;
    new_dir_stream << std::string(cwd) << "/" << subdir;
    std::string new_dir = new_dir_stream.str();

    struct stat info;
    if (stat(new_dir.c_str(), &info) != 0) {
        // Directory does not exist, create it
        if (mkdir(new_dir.c_str(), 0755) != 0) {
            std::cerr << "mkdir() error: " << strerror(errno) << std::endl;
            exit(1);
        } else {
            std::cout << "Directory created: " << new_dir << std::endl;
        }
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << new_dir << " is not a directory." << std::endl;
        exit(1);
    }

    // Change the working directory to the relative path
    if (chdir(subdir.c_str()) != 0) {
        std::cerr << "chdir() error: " << strerror(errno) << std::endl;
        exit(1);
    }
}

