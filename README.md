# 2D-Graphics-Project
Hosts my project code for 2D Graphics

C Bitmap Processing Program by Colin Tomcak

Only works on Unix-like systems due to pthreads.
It was initially multi-platform because it only uses the C standard library, but I wanted to add multi-threading and I used POSIX threads (pthread.h).

Only works on Bitmap files with 24bit RGB (most BMPs).
This is also because I only wanted to use libc, and writing stuff to interpret complex compressed file formats is no trivial task.

Compiler flags I used: `gcc -Wall -Wextra -Wpedantic -Werror -Ofast -o image image.c -lpthread`

Didn't have time to add user input, so enter the image name in the defined macro field, and uncomment the functions you want to use in the outlined section in main().
