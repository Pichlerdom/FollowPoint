#c compiler
CC = g++

#cuda compiler
NVCC = nvcc 

# define c++ compiler flags
CFLAGS =-g -Wall $(SDL2CFLAGS)

#define cuda compiler flags
NVCFLAGS =-g -lineinfo -maxrregcount=0 -gencode=arch=compute_50,code=\"sm_50,compute_50\" $(SDL2CFLAGS)

#define library paths
LFLAGS = -L"/usr/lib/x86_64-linux-gnu" 

#define library names
LIBS = -lcuda -lcudart -lSDL2 -lSDL2_image -lm -lpthread -lv4lconvert

#define SDL2 Options
SDL2CFLAGS = -D_REENRRANT
SDL2INCLUDES = -I/usr/include/SDL2 

#define include directories
INCLUDES =-I../include $(SDL2INCLUDES) -I../libv4l/include

#define object files
OBJS = $(SRCS:.cpp=.o) $(NVSRCS:.cu=.o)

# define the C source files
SRCS = main.cpp webcam_interface.cpp follow_point_display.cpp serial_interface.cpp obj_detection.cpp 

# define the cuda source files
NVSRCS = obj_detection_cuda.cu obj_detection_kernel.cu

# define the executable file 
MAIN = ../bin/followPoint

# add .cu suffix so we can compile cuda files
.SUFFIXES :.cu .cpp

.PHONY:all clean 

all:$(MAIN)


$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(INCLUDES) $(LFLAGS) $(LIBS) $(SDLCONF)

%.o:%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ 
	
%.o:%.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm *.o