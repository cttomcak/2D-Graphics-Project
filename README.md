# 2D-Graphics-Project
Hosts my project code for 2D Graphics

C Image Processing Program by Colin Tomcak

Only works on Unix-like systems due to pthreads.
It was initially multi-platform because it only uses the C standard library, but I wanted to add multi-threading and I used POSIX threads (pthreads).

Only works on Bitmap files with 24bit RGB (Almost all BMPs).
This is also because I only wanted to use libc, and writing stuff to interpret other file formats sounds very hard. ¯\\_(ツ)_/¯

Recommended compiler flags: gcc -Wall -Wextra -Wpedantic -Werror -Ofast -o image image.c -lpthread