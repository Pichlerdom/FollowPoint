
#include "followPoint.h"
#include "webcam_interface.h"
#include "obj_detection.h"
#include "serial_interface.h"
#include "obj_detection_cuda.h"

#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <linux/videodev2.h>
#include <libv4lconvert.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))

static int xioctl(int fd, int request, void *arg)
{
    int r;
	do r = ioctl (fd, request, arg);
	while (-1 == r && EINTR == errno);
	return r;
}

int allocCamera(ImgCont *imgCont, uint8_t **framebuf){
	struct v4l2_capability caps = {0};
	struct v4l2_format fmt = {0};
	struct v4l2_buffer buf = {0};

	int fd_cam = open(DEFAULT_VIDEO_DEVICE, O_RDWR);

	if(-1 == xioctl(fd_cam, VIDIOC_QUERYCAP, &caps)){
		perror("VIDIOC_QUERYCAP");
		exit(EXIT_FAILURE);
	}

	if(!(caps.capabilities & V4L2_CAP_STREAMING)){
		fprintf(stderr, "The device does not handle single-planar video capture.\n");
		exit(1);
	}
	
	fmt.type				= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width		= WEBCAM_FRAME_WIDTH;
	fmt.fmt.pix.height		= WEBCAM_FRAME_HEIGHT;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_OV518;
	fmt.fmt.pix.field		= V4L2_FIELD_INTERLACED;

	printf("%d\n",fmt.fmt.pix.pixelformat);
	if (-1 == xioctl(fd_cam, VIDIOC_S_FMT, &fmt)){
		perror("Setting Pixel Format");
		return 1;
	}	
	
	if (-1 == xioctl(fd_cam, VIDIOC_G_FMT, &fmt)){
		perror("Setting Pixel Format");
		return 1;
	}
	printf("%d\n",fmt.fmt.pix.pixelformat);

	struct v4l2_requestbuffers req = {0};
	req.count = 1;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;
 
	if (-1 == xioctl(fd_cam, VIDIOC_REQBUFS, &req))
	{
		perror("Requesting Buffer");
		return 1;
	}
	
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;

	if(-1 == xioctl(fd_cam, VIDIOC_QUERYBUF, &buf))
	{
		perror("Querying Buffer");
		return 1;
	}

	*framebuf = (uint8_t*) mmap (NULL, buf.length,
					PROT_READ | PROT_WRITE,	
					MAP_SHARED, 
					fd_cam, buf.m.offset);
	


	if(-1 == xioctl(fd_cam, VIDIOC_QBUF, &buf))
	{
		printf("Retrieving Frame");
		return 1;
	}

	if(-1 == xioctl(fd_cam, VIDIOC_STREAMON, &(buf.type)))	
	{
		perror("Start Capture");
		return 1;
	}

	return fd_cam;
	

}

void closeCamera(int fd_cam){
	close(fd_cam);
}


void *main_capture_loop(void *arg){
	ImgCont *imgCont = (ImgCont*) arg;

	uint8_t *framebuf;
	int fd_camera = allocCamera(imgCont, &framebuf);

	int curTime = SDL_GetTicks();
	int deltaTime = 0;

	int fd_serial = open_com_port("/dev/ttyACM0");

	while(imgCont->control.capture){

		curTime = SDL_GetTicks();

		readframe(imgCont, fd_camera, framebuf);

		write_delta(fd_serial, imgCont->laser.dx, imgCont->laser.dy);

		deltaTime = SDL_GetTicks() - curTime;
		printf("capture time %d \n", deltaTime); 
	}
	

	closeCamera(fd_camera);
	return NULL;

}

void writeframebuf(int fd, uint8_t *framebuf,int framebufsize, ImgCont *imgCont){
	struct v4l2_format dest_fmt, src_fmt;
	CLEAR(dest_fmt);
	CLEAR(src_fmt);
	
	src_fmt.type				= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	src_fmt.fmt.pix.width		= WEBCAM_FRAME_WIDTH;
	src_fmt.fmt.pix.height		= WEBCAM_FRAME_HEIGHT;
	src_fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_OV518;
	src_fmt.fmt.pix.field		= V4L2_FIELD_INTERLACED;


	dest_fmt.type				= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	dest_fmt.fmt.pix.width		= WEBCAM_FRAME_WIDTH;
	dest_fmt.fmt.pix.height		= WEBCAM_FRAME_HEIGHT;
	dest_fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
	dest_fmt.fmt.pix.field		= V4L2_FIELD_INTERLACED;

	struct v4lconvert_data *conv_data = v4lconvert_create(fd);
	int byteswriten = v4lconvert_convert(	conv_data,
						&src_fmt, &dest_fmt,
						framebuf, framebufsize, 
						imgCont->img, WEBCAM_FRAME_WIDTH * WEBCAM_FRAME_HEIGHT * 3 * sizeof(uint8_t));	
	printf("byteswriten:%d\n",byteswriten);
	v4lconvert_destroy(conv_data);

}

void readframe(ImgCont *imgCont, int fd_camera, uint8_t *framebuf){
	struct v4l2_buffer buf;
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;



	if(-1 == xioctl(fd_camera, VIDIOC_DQBUF, &buf))
	{
		perror("VIDIOC_QDBUF");
		return ;
	}

	pthread_mutex_lock(&(imgCont->imgMtx));	

	printf("bytesused: %d\n",buf.bytesused);
	writeframebuf(fd_camera, framebuf, buf.bytesused, imgCont);

	if(imgCont->control.detection){
		cuda_test(imgCont,imgCont->comp);
	}
	
	pthread_mutex_unlock(&(imgCont->imgMtx));

	if(-1 == xioctl(fd_camera , VIDIOC_QBUF, &buf))
	{
		perror("VIDIOC_QBUF");
		return ;
	}
}
