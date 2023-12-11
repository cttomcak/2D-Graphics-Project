/* C Image Processing Program by Colin Tomcak */

/* Only works on Unix-like systems due to pthreads.
        It was initially multi-platform because it only uses
        the C standard library, but I wanted to add multi-threading
        and I used POSIX threads (pthreads). */

/* Only works on Bitmap files with 24bit RGB (Almost all BMPs).
        This is also because I only wanted to use libc, and writing
        stuff to interpret other file formats sounds very hard. 
        ¯\_(ツ)_/¯ */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>

/* Recommended compiler flags:
        gcc -Wall -Wextra -Wpedantic -Werror -Ofast -o image image.c -lpthread */

/* In main(), you can change which image processing functions run */

/* Constants */
#define FILE_IN_NAME "example.bmp"
#define FILE_OUT_NAME "out.bmp"
#define HEADER_SIZE 54
#define MAX_COLOR 255
#define NUM_THREADS 20
#define NANO_IN_SECOND 1.0E9
#define BITS_PER_PIXEL 24

/* Set this to 0 to not write an output file (for time testing) */
#define DO_WRITE_FILE 1

/* Constants for some functions (higher weight means more effect) */
/* You can actually set the any of these weights negative to acomplish
        opposite effect, but I chose to have separate functions anyways
        (i.e., setting SATURATE_WEIGHT negative will desaturate) */
#define SATURATE_WEIGHT 0.5f
#define DESATURATE_WEIGHT 0.5f
#define BRIGHTEN_WEIGHT 50
#define DARKEN_WEIGHT 50

/* Cutoff thresholds for the set_dim.. and set_bright.. functions */
#define HIGH_PASS_THRESHOLD 60
#define LOW_PASS_THRESHOLD 200

/* Mini Functions */
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))
#define MAX(a,b) (((a)>(b)) ? (a):(b))

/* Image Processing Kernels/Matrices */
#define IDENTITY_KERNEL {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}
#define BOX_BLUR_KERNEL {{.11, .11, .11}, {.11, .12, .11}, {.11, .11, .11}}
#define GAUSSIAN_BLUR_KERNEL {{.0625, .125, .0625}, {.125, .25, .125}, {.0625, .125, .0625}}
#define SHARPEN_KERNEL {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}}
#define EMBOSS_KERNEL {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}}
#define EDGE_DETECT_KERNEL {{-2, -2, -2}, {-2, 16, -2}, {-2, -2, -2}}
#define SOBEL_B_KERNEL {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}
#define SOBEL_T_KERNEL {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}}
#define SOBEL_L_KERNEL {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}}
#define SOBEL_R_KERNEL {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}

/* Struct to store pixel info in array (3 bytes) */
typedef struct pixel_info
{
    uint8_t blue;
    uint8_t green;
    uint8_t red;
} pixel_info;

/* Struct to store info about the image (16 bytes) */
typedef struct image_info
{
    int width;
    int height;
    pixel_info *pixel_data;
} image_info;

/* Struct to pass info for threading (40 bytes) */
typedef struct thread_info
{
    image_info *i_info;
    int start_y;
    int end_y;
    pixel_info **pixel_array;
    pixel_info **new_pixel_array;
    double (*kernel)[3];
} thread_info;

/* Global variables */
FILE *fileIN = NULL;
FILE *fileOUT = NULL;
pixel_info *global_pixel_data = NULL;

/* Cleanup function for globals */
void cleanup(void);

/* If the user presses CTRL+C we can do graceful cleanup */
void SIGINT_handler(int sig);

/* Converts image to greyscale (in memory) */
void greyscale(image_info *info);

/* Inverts image (in memory) */
void invert(image_info *info);

/* Saturates image (in memory) */
void saturate(image_info *info);

/* Desaturates image (in memory) */
void desaturate(image_info *info);

/* Brightens, image (in memory) */
void brighten(image_info *info);

/* Darkens image (in memory) */
void darken(image_info *info);

/* Sets pixels below a brightness threshold to black (in memory) */
void set_dim_to_black(image_info *info);

/* Sets pixels above a brightness threshold to white (in memory) */
void set_bright_to_white(image_info *info);

/* Leaves only the red color in the image (in memory) */
void red_only(image_info *info);

/* Leaves only the green color in the image (in memory) */
void green_only(image_info *info);

/* Leaves only the blue color in the image (in memory) */
void blue_only(image_info *info);

/* Swaps red and green in image (in memory) */
void swap_r_and_g(image_info *info);

/* Swaps red and blue in image (in memory) */
void swap_r_and_b(image_info *info);

/* Swaps green and blue in image (in memory) */
void swap_g_and_b(image_info *info);

/* Generalized convolve function that uses a 3x3 kernel (new memory) */
pixel_info* convolve(image_info *info, double kernel[3][3]);

/* Multi-threading helper function for the convolve function */
void* convolve_threader(void *t_info);

/* Uses the identity kernel for testing (new memory) */
void identity(image_info *info);

/* Does box blur using kernel (new memory) */
void box_blur(image_info *info);

/* Does gaussian blur using kernel (new memory) */
void gaussian_blur(image_info *info);

/* Sharpens image using kernel (new memory) */
void sharpen(image_info *info);

/* Embosses image using kernel (new memory) */
void emboss(image_info *info);

/* Does simple edge detection using kernel (new memory) */
void simple_edge_detection(image_info *info);

/* Does much more complex edge detection (much new memory)
        This is not the full canny algorithm, just the first part */
void canny_edge_detection(image_info *info);

/* Opens the input file to the global fileIN variable */
void open_global_file_in(void);

/* Reads the header of the file into an array, which is 54 bytes */
void read_file_header(uint8_t *header);

/* Checks to try and make sure we're working with a bitmap file
        Also checks to make sure the bitmap has exactly 24 bits per pixel */
void check_file_and_bpp(uint8_t *header);

/* Read the image's pixel data into the global pixel data buffer */
void read_global_pixel_data(size_t image_size);

/* Opens the output file to the global fileOUT variable */
void open_global_file_out(void);

/* Writes the file header using the array, which is 54 bytes */
void write_file_header(uint8_t *header);

/* Writes the global pixel data buffer to the output file */
void write_global_pixel_data(size_t image_size);

/* Main */
int main(void)
{
    /* For measuring the real runtime of the program */
    struct timespec start, lap, end;
    double elapsed;

    uint8_t header[HEADER_SIZE];

    int32_t image_width, image_height;
    size_t image_size;

    /* Installs the SIGINT handler. This means that if the user presses CTRL+C, the SIGINT
            handler function will run, which exits the program little more gracefully */
    signal(SIGINT, SIGINT_handler);

    clock_gettime(CLOCK_MONOTONIC, &start);
    clock_gettime(CLOCK_MONOTONIC, &lap);

    open_global_file_in();
    read_file_header(header);
    check_file_and_bpp(header);

    /* Gets information about the image at these specific locations in the header, and prints it */
    image_width = *(int32_t*)&header[18];
	image_height = *(int32_t*)&header[22];
    image_size = (size_t)(image_width*image_height);
	printf("Image size (WxH): %" PRId32 "x%" PRId32 ".\n", image_width, image_height);

    read_global_pixel_data(image_size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - lap.tv_sec);
    elapsed += (end.tv_nsec - lap.tv_nsec) / NANO_IN_SECOND;
    printf("Time to read file: \t%.4lf seconds.\n", elapsed);
    clock_gettime(CLOCK_MONOTONIC, &lap);

    /* Declare an image info struct; it's easy to manage parameters this way */
    image_info i_info = {image_width, image_height, global_pixel_data};
    image_info *info = &i_info;

    /* Start Image Processing
            This section of the program is the only place where memory
            is potentially allocated for more pixel data buffers */

    // greyscale(info);
    // invert(info);
    // saturate(info);
    // desaturate(info);
    // brighten(info);
    // darken(info);
    // set_dim_to_black(info);
    // set_bright_to_white(info);
    // red_only(info);
    // green_only(info);
    // blue_only(info);
    // swap_r_and_g(info);
    // swap_r_and_b(info);
    // swap_g_and_b(info);
    // identity(info);
    // box_blur(info);
    // gaussian_blur(info);
    // sharpen(info);
    // emboss(info);
    // simple_edge_detection(info);
    canny_edge_detection(info);

    /* End Image Processing 
            The only allocated memory past this point is the original
            global pixel data buffer, which is freed in cleanup() */

    global_pixel_data = info->pixel_data;

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - lap.tv_sec);
    elapsed += (end.tv_nsec - lap.tv_nsec) / NANO_IN_SECOND;
    printf("Time for processing: \t%.4lf seconds.\n", elapsed);
    clock_gettime(CLOCK_MONOTONIC, &lap);

    /* Choose whether or not to write the file. This conditional is for
            when I'm testing the program's speed on massive images and
            don't want to wear out my SSD with constant 100MB writes */
    if (DO_WRITE_FILE)
    {
        open_global_file_out();
        write_file_header(header);
        write_global_pixel_data(image_size);

        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - lap.tv_sec);
        elapsed += (end.tv_nsec - lap.tv_nsec) / NANO_IN_SECOND;
        printf("Time to write file: \t%.4lf seconds.\n", elapsed);
    }

    elapsed = (end.tv_sec - start.tv_sec);
    elapsed += (end.tv_nsec - start.tv_nsec) / NANO_IN_SECOND;
    printf("Total program time: \t%.4lf seconds.\n", elapsed);

    cleanup();
    exit(EXIT_SUCCESS);
}

void cleanup(void)
{
    if (fileIN != NULL)
    {
        fclose(fileIN);
        fileIN = NULL;
    }

    if (fileOUT != NULL)
    {
        fclose(fileOUT);
        fileOUT = NULL;
    }

    if (global_pixel_data != NULL)
    {
        free(global_pixel_data);
        global_pixel_data = NULL;
    }
}

void SIGINT_handler(int sig)
{
    printf("\nProgram interrupted (%d). It will now be terminated.\n", sig);
    cleanup();
    exit(EXIT_FAILURE);
}

void greyscale(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    uint8_t average;
    for (int i = 0; i < image_size; i++) {
        average = (uint8_t)((pixel_data[i].red + pixel_data[i].green + pixel_data[i].blue)/3);
        pixel_data[i].red = average;
        pixel_data[i].green = average;
        pixel_data[i].blue = average;
    }
}

void invert(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    for (int i = 0; i < image_size; i++) {
        pixel_data[i].red = MAX_COLOR - pixel_data[i].red;
        pixel_data[i].green = MAX_COLOR - pixel_data[i].green;
        pixel_data[i].blue = MAX_COLOR - pixel_data[i].blue;
    }
}

void saturate(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    uint8_t average;
    int r, g, b;
    for (int i = 0; i < image_size; i++) {
        average = (uint8_t)((pixel_data[i].red + pixel_data[i].green + pixel_data[i].blue)/3);
        r = (int)pixel_data[i].red;
        g = (int)pixel_data[i].green;
        b = (int)pixel_data[i].blue;
        r += (int)(SATURATE_WEIGHT*(float)(r-average));
        g += (int)(SATURATE_WEIGHT*(float)(g-average));
        b += (int)(SATURATE_WEIGHT*(float)(b-average));
        if (r > MAX_COLOR) {r = MAX_COLOR;}
        if (g > MAX_COLOR) {g = MAX_COLOR;}
        if (b > MAX_COLOR) {b = MAX_COLOR;}
        if (r < 0) {r = 0;}
        if (g < 0) {g = 0;}
        if (b < 0) {b = 0;}
        pixel_data[i].red = (uint8_t)r;
        pixel_data[i].green = (uint8_t)g;
        pixel_data[i].blue = (uint8_t)b;
    }
}

void desaturate(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    uint8_t average;
    int r, g, b;
    for (int i = 0; i < image_size; i++) {
        average = (uint8_t)((pixel_data[i].red + pixel_data[i].green + pixel_data[i].blue)/3);
        r = (int)pixel_data[i].red;
        g = (int)pixel_data[i].green;
        b = (int)pixel_data[i].blue;
        r -= (int)(DESATURATE_WEIGHT*(float)(r-average));
        g -= (int)(DESATURATE_WEIGHT*(float)(g-average));
        b -= (int)(DESATURATE_WEIGHT*(float)(b-average));
        if (r > MAX_COLOR) {r = MAX_COLOR;}
        if (g > MAX_COLOR) {g = MAX_COLOR;}
        if (b > MAX_COLOR) {b = MAX_COLOR;}
        if (r < 0) {r = 0;}
        if (g < 0) {g = 0;}
        if (b < 0) {b = 0;}
        pixel_data[i].red = (uint8_t)r;
        pixel_data[i].green = (uint8_t)g;
        pixel_data[i].blue = (uint8_t)b;
    }
}

void brighten(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    int r, g, b;
    for (int i = 0; i < image_size; i++) {
        r = (int)pixel_data[i].red + BRIGHTEN_WEIGHT;
        g = (int)pixel_data[i].green + BRIGHTEN_WEIGHT;
        b = (int)pixel_data[i].blue + BRIGHTEN_WEIGHT;
        if (r > MAX_COLOR) {r = MAX_COLOR;}
        if (g > MAX_COLOR) {g = MAX_COLOR;}
        if (b > MAX_COLOR) {b = MAX_COLOR;}
        if (r < 0) {r = 0;}
        if (g < 0) {g = 0;}
        if (b < 0) {b = 0;}
        pixel_data[i].red = (uint8_t)r;
        pixel_data[i].green = (uint8_t)g;
        pixel_data[i].blue = (uint8_t)b;
    }
}

void darken(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    int r, g, b;
    for (int i = 0; i < image_size; i++) {
        r = (int)pixel_data[i].red - DARKEN_WEIGHT;
        g = (int)pixel_data[i].green - DARKEN_WEIGHT;
        b = (int)pixel_data[i].blue - DARKEN_WEIGHT;
        if (r > MAX_COLOR) {r = MAX_COLOR;}
        if (g > MAX_COLOR) {g = MAX_COLOR;}
        if (b > MAX_COLOR) {b = MAX_COLOR;}
        if (r < 0) {r = 0;}
        if (g < 0) {g = 0;}
        if (b < 0) {b = 0;}
        pixel_data[i].red = (uint8_t)r;
        pixel_data[i].green = (uint8_t)g;
        pixel_data[i].blue = (uint8_t)b;
    }
}

void set_dim_to_black(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    uint8_t average;
    for (int i = 0; i < image_size; i++) {
        average = (uint8_t)((pixel_data[i].red + pixel_data[i].green + pixel_data[i].blue)/3);
        if (average < HIGH_PASS_THRESHOLD)
        {
            pixel_data[i].red = 0;
            pixel_data[i].green = 0;
            pixel_data[i].blue = 0;
        }
    }
}

void set_bright_to_white(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    uint8_t average;
    for (int i = 0; i < image_size; i++) {
        average = (uint8_t)((pixel_data[i].red + pixel_data[i].green + pixel_data[i].blue)/3);
        if (average > LOW_PASS_THRESHOLD)
        {
            pixel_data[i].red = MAX_COLOR;
            pixel_data[i].green = MAX_COLOR;
            pixel_data[i].blue = MAX_COLOR;
        }
    }
}

void red_only(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    for (int i = 0; i < image_size; i++) {
        pixel_data[i].green = 0;
        pixel_data[i].blue = 0;
    }
}

void green_only(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    for (int i = 0; i < image_size; i++) {
        pixel_data[i].red = 0;
        pixel_data[i].blue = 0;
    }
}

void blue_only(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    for (int i = 0; i < image_size; i++) {
        pixel_data[i].red = 0;
        pixel_data[i].green = 0;
    }
}

void swap_r_and_g(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    uint8_t temp;
    for (int i = 0; i < image_size; i++) {
        temp = pixel_data[i].red;
        pixel_data[i].red = pixel_data[i].green;
        pixel_data[i].green = temp;
    }
}

void swap_r_and_b(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    uint8_t temp;
    for (int i = 0; i < image_size; i++) {
        temp = pixel_data[i].red;
        pixel_data[i].red = pixel_data[i].blue;
        pixel_data[i].blue = temp;
    }
}

void swap_g_and_b(image_info *info)
{
    pixel_info *pixel_data = info->pixel_data;
    int image_size = info->width * info->height;
    uint8_t temp;
    for (int i = 0; i < image_size; i++) {
        temp = pixel_data[i].green;
        pixel_data[i].green = pixel_data[i].blue;
        pixel_data[i].blue = temp;
    }
}

pixel_info* convolve(image_info *info, double kernel[3][3])
{
    int image_width = info->width;
    int image_height = info->height;

    size_t image_size = (size_t)(image_width * image_height);
    pixel_info *new_pixel_data = (pixel_info*)malloc(sizeof(pixel_info)*image_size);
    if (new_pixel_data == NULL)
    {
        printf("ERROR:  Failed to allocate memory for pixel data.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    pixel_info *pixel_data = info->pixel_data;
    pixel_info *pixel_array[image_height];
    for (int i = 0; i < image_height; i++)
    {
        pixel_array[i] = pixel_data + (i * image_width);
    }

    pixel_info *new_pixel_array[image_height];
    for (int i = 0; i < image_height; i++)
    {
        new_pixel_array[i] = new_pixel_data + (i * image_width);
    }

    pthread_t threads[NUM_THREADS];
    thread_info info_structs[NUM_THREADS];
    for (int i = 0; i < (NUM_THREADS - 1); i++)
    {
        info_structs[i].i_info = info;
        info_structs[i].start_y = image_height/NUM_THREADS * i;
        info_structs[i].end_y = image_height/NUM_THREADS * (i + 1);
        info_structs[i].pixel_array = pixel_array;
        info_structs[i].new_pixel_array = new_pixel_array;
        info_structs[i].kernel = kernel;
        pthread_create(threads + i, NULL, convolve_threader, (void*)(info_structs + i));
    }
    int idx = NUM_THREADS - 1;
    info_structs[idx].i_info = info;
    info_structs[idx].start_y = image_height/NUM_THREADS * idx;
    info_structs[idx].end_y = image_height;
    info_structs[idx].pixel_array = pixel_array;
    info_structs[idx].new_pixel_array = new_pixel_array;
    info_structs[idx].kernel = kernel;
    pthread_create(threads + idx, NULL, convolve_threader, (void*)(info_structs + idx));

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    return new_pixel_data;
}

void* convolve_threader(void *t_info)
{
    thread_info *info = (thread_info*)t_info;

    int image_height, image_width, r, g, b, radius,
    m_y, m_x, f_x, f_y, x, y, xx, yy, start, end;

    image_width = info->i_info->width;
    image_height = info->i_info->height;

    pixel_info **pixel_array = info->pixel_array;
    pixel_info **new_pixel_array = info->new_pixel_array;

    start = info->start_y;
    end = info->end_y;

    double (*kernel)[3] = info->kernel;

    radius = 1;
    for (y = start; y < end; y++)
    {
        for (x = 0; x < image_width; x++)
        {
            r = g = b = 0;
            for (yy = y-radius; yy <= y+radius; yy++)
            {
                for (xx = x-radius; xx <= x+radius; xx++)
                {
                    f_x = xx;
                    f_y = yy;

                    if (f_x < 0) {f_x *= -1;}
                    if (f_y < 0) {f_y *= -1;}
                    if (f_x >= image_width) {f_x -= (f_x - image_width + 1)*2;}
                    if (f_y >= image_height) {f_y -= (f_y - image_height + 1)*2;}

                    m_y = y - yy + radius;
                    m_x = xx - x + radius;
                    r += (int)(pixel_array[f_y][f_x].red * kernel[m_y][m_x]);
                    g += (int)(pixel_array[f_y][f_x].green * kernel[m_y][m_x]);
                    b += (int)(pixel_array[f_y][f_x].blue * kernel[m_y][m_x]);
                }
            }
            if (r > MAX_COLOR) {r = MAX_COLOR;}
            if (g > MAX_COLOR) {g = MAX_COLOR;}
            if (b > MAX_COLOR) {b = MAX_COLOR;}
            if (r < 0) {r = 0;}
            if (g < 0) {g = 0;}
            if (b < 0) {b = 0;}
            new_pixel_array[y][x].red = (uint8_t)r;
            new_pixel_array[y][x].green = (uint8_t)g;
            new_pixel_array[y][x].blue = (uint8_t)b;
        }
    }
    return NULL;
}

void identity(image_info *info)
{
    double identity_kernel[3][3] = IDENTITY_KERNEL;
    pixel_info *identity_pd = convolve(info, identity_kernel);
    free(info->pixel_data);
    info->pixel_data = identity_pd;
}

void box_blur(image_info *info)
{
    double box_blur_kernel[3][3] = BOX_BLUR_KERNEL;
    pixel_info *box_blured_pd = convolve(info, box_blur_kernel);
    free(info->pixel_data);
    info->pixel_data = box_blured_pd;
}

void gaussian_blur(image_info *info)
{
    double gauss_blur_kernel[3][3] = GAUSSIAN_BLUR_KERNEL;
    pixel_info *gauss_blured_pd = convolve(info, gauss_blur_kernel);
    free(info->pixel_data);
    info->pixel_data = gauss_blured_pd;
}

void sharpen(image_info *info)
{
    double sharpen_kernel[3][3] = SHARPEN_KERNEL;
    pixel_info *sharpened_pd = convolve(info, sharpen_kernel);
    free(info->pixel_data);
    info->pixel_data = sharpened_pd;
}

void emboss(image_info *info)
{
    double emboss_kernel[3][3] = EMBOSS_KERNEL;
    pixel_info *embossed_pd = convolve(info, emboss_kernel);
    free(info->pixel_data);
    info->pixel_data = embossed_pd;
}

void simple_edge_detection(image_info *info)
{
    greyscale(info);
    gaussian_blur(info);

    double edge_detect_kernel[3][3] = EDGE_DETECT_KERNEL;
    pixel_info *edge_detected_pd = convolve(info, edge_detect_kernel);
    free(info->pixel_data);
    info->pixel_data = edge_detected_pd;
    
    set_dim_to_black(info);
}

void canny_edge_detection(image_info *info)
{
    greyscale(info);
    gaussian_blur(info);

    int image_width = info->width;
    int image_height = info->height;

    double sobel_t_kernel[3][3] = SOBEL_T_KERNEL;
    pixel_info *sobel_t_pd = convolve(info, sobel_t_kernel);

    double sobel_b_kernel[3][3] = SOBEL_B_KERNEL;
    pixel_info *sobel_b_pd = convolve(info, sobel_b_kernel);

    double sobel_l_kernel[3][3] = SOBEL_L_KERNEL;
    pixel_info *sobel_l_pd = convolve(info, sobel_l_kernel);

    double sobel_r_kernel[3][3] = SOBEL_R_KERNEL;
    pixel_info *sobel_r_pd = convolve(info, sobel_r_kernel);

    int image_size = image_width * image_height;

    pixel_info *pixel_data = info->pixel_data;
    uint8_t r, g, b;
    for (int i = 0; i < image_size; i++)
    {
        r = MAX(MAX(sobel_t_pd[i].red, sobel_b_pd[i].red), MAX(sobel_l_pd[i].red, sobel_r_pd[i].red));
        g = MAX(MAX(sobel_t_pd[i].green, sobel_b_pd[i].green), MAX(sobel_l_pd[i].green, sobel_r_pd[i].green));
        b = MAX(MAX(sobel_t_pd[i].blue, sobel_b_pd[i].blue), MAX(sobel_l_pd[i].blue, sobel_r_pd[i].blue));

        pixel_data[i].red = r;
        pixel_data[i].green = g;
        pixel_data[i].blue = b;
    }

    free(sobel_t_pd);
    free(sobel_b_pd);
    free(sobel_l_pd);
    free(sobel_r_pd);

    set_dim_to_black(info);
}

void open_global_file_in(void)
{
    fileIN = fopen(FILE_IN_NAME, "r");
    if (fileIN == NULL)
    {
        printf("ERROR:  Cannot open input file.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
}

void read_file_header(uint8_t *header)
{
    size_t bytes_read = fread(header, sizeof(uint8_t), (size_t)HEADER_SIZE, fileIN);
    if (bytes_read != HEADER_SIZE)
    {
        printf("ERROR:  Cannot read header from file.\n");
        printf("\tBytes read from header: %ld\n", bytes_read);
        printf("\tWas end of file bool: %s\n", feof(fileIN) ? "true" : "false");
        printf("\tWas error bool: %s\n", ferror(fileIN) ? "true" : "false");
        cleanup();
        exit(EXIT_FAILURE);
    }
}

void check_file_and_bpp(uint8_t *header)
{
    if ('B' != (char)header[0] || 'M' != (char)header[1])
    {
        printf("ERROR:  Not a bitmap file.\n");
        printf("\tThe first 2 bytes of the header should be 'BM'\n");
        printf("\tInstead, they are: '%c%c'\n", (char)header[0], (char)header[1]);
        cleanup();
        exit(EXIT_FAILURE);
    }

    int16_t bits_per_pixel = *(uint16_t*)&header[28];
    if (bits_per_pixel != BITS_PER_PIXEL)
    {
        printf("ERROR:  Bits per pixel is not %d.\n", BITS_PER_PIXEL);
        printf("\tYour image's bits per pixel: %" PRId16 "\n", bits_per_pixel);
        printf("\tMy program isn't written to handle this case. Sorry.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
}

void read_global_pixel_data(size_t image_size)
{
    global_pixel_data = (pixel_info*)malloc(sizeof(pixel_info)*image_size);
    if (global_pixel_data == NULL)
    {
        printf("ERROR:  Failed to allocate memory for pixel data.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    size_t bytes_read = fread(global_pixel_data, sizeof(pixel_info), image_size, fileIN);
    if (bytes_read != image_size)
    {
        printf("ERROR:  Cannot read pixel data from file.\n");
        printf("\tBytes read from pixel data: %ld\n", bytes_read);
        printf("\tWas end of file bool: %s\n", feof(fileIN) ? "true" : "false");
        printf("\tWas error bool: %s\n", ferror(fileIN) ? "true" : "false");
        cleanup();
        exit(EXIT_FAILURE);
    }
}

void open_global_file_out(void)
{
    fileOUT = fopen(FILE_OUT_NAME, "w");
    if (fileOUT == NULL)
    {
        printf("ERROR:  Cannot open output file.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
}

void write_file_header(uint8_t *header)
{
    size_t bytes_written = fwrite(header, sizeof(uint8_t), (size_t)HEADER_SIZE, fileOUT);
    if (bytes_written != HEADER_SIZE)
    {
        printf("ERROR:  Cannot write header to file.\n");
        printf("\tBytes written from header: %ld\n", bytes_written);
        cleanup();
        exit(EXIT_FAILURE);
    }
}

void write_global_pixel_data(size_t image_size)
{
    size_t bytes_written = fwrite(global_pixel_data, sizeof(pixel_info), image_size, fileOUT);
    if (bytes_written != image_size)
    {
        printf("ERROR:  Cannot write pixel data to file.\n");
        printf("\tBytes written from pixel data: %ld\n", bytes_written);
        cleanup();
        exit(EXIT_FAILURE);
    }
}