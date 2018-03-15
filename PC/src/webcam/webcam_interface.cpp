
#include "webcam_interface.h"
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <linux/videodev2.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))

#define WEBCAM_FRAMERATE 30

//file_name		->		Name of video camera file
//widht:height	->		open_camera will try to set the frame size to width:height.
//						If width:height is not supported by camera width:height will
//						be set to the closest supported value.
//frame_buf		->		Buffer that holdes the video data in yuyv420 will be used by fill_frame_buf.
//						and convert_YUYV_to_RGB.
//						Will be allocated by open_camera.
//frame_buf_size->		Size of the frame buffer. Set by open_camera. 
// returns the file handle
// 1 on error
int open_camera(char *file_name, int *width, int *height,uint8_t **frame_buf, int *frame_buf_size){
	struct v4l2_capability caps = {0};
	struct v4l2_format fmt = {0};
	struct v4l2_buffer buf = {0};

	int fd_cam = open(file_name, O_RDWR);

	if(-1 == xioctl(fd_cam, VIDIOC_QUERYCAP, &caps)){
		perror("VIDIOC_QUERYCAP");
		exit(EXIT_FAILURE);
	}

	if(!(caps.capabilities & V4L2_CAP_STREAMING)){
		fprintf(stderr, "The device does not handle single-planar video capture.\n");
		exit(1);
	}
	
	fmt.type			= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width		= *width;
	fmt.fmt.pix.height		= *height;
	fmt.fmt.pix.pixelformat 	= V4L2_PIX_FMT_YUYV;
	fmt.fmt.pix.field		= V4L2_FIELD_NONE;

	if (-1 == xioctl(fd_cam, VIDIOC_S_FMT, &fmt)){
		perror("Setting Pixel Format");
		return 1;
	}	
	
	if (-1 == xioctl(fd_cam, VIDIOC_G_FMT, &fmt)){
		perror("Setting Pixel Format");
		return 1;
	}
	
	if(!set_framerate(fd_cam, WEBCAM_FRAMERATE)){
		perror("Could not set framerate");
	}	


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

	*frame_buf = (uint8_t*) mmap (NULL, buf.length,
					PROT_READ | PROT_WRITE,	
					MAP_SHARED, 
					fd_cam, buf.m.offset);
	
	*frame_buf_size = buf.length;


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

void close_camera(int fd_camera){
	close(fd_camera);
}

void fill_frame_buf_yuyv(int fd_camera, uint8_t *frame_buf){

	struct v4l2_buffer buf;
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;



	if(-1 == xioctl(fd_camera, VIDIOC_DQBUF, &buf))
	{
		perror("VIDIOC_QDBUF");
		return ;
	}

	if(-1 == xioctl(fd_camera , VIDIOC_QBUF, &buf))
	{
		perror("VIDIOC_QBUF");
		return ;
	}
}

void fill_read_frame_rgb(int fd_camera, uint8_t *frame_buf, uint8_t *imgrgb, int width, int height){
	
	struct v4l2_buffer buf;
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;

	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(fd_camera, &fds);
	struct timeval tv = {0};
	tv.tv_sec = 2;
	int r = select(fd_camera + 1, &fds, NULL, NULL, &tv);
	if (-1 == r) {
		perror("Waiting for Frame");
		return ;
	}


	if(-1 == xioctl(fd_camera, VIDIOC_DQBUF, &buf))
	{
		perror("VIDIOC_QDBUF");
		return ;
	}

	convert_YUYV_to_RGB(frame_buf, imgrgb, width,height);
	

	
	if(-1 == xioctl(fd_camera , VIDIOC_QBUF, &buf))
	{
		perror("VIDIOC_QBUF");
		return ;
	}

}

//converts a YUYV4:2:0 img to rgb
void convert_YUYV_to_RGB(uint8_t *imgyuyv, uint8_t *imgrgb, int witdh, int 
		height){
	for(int i = 0; i < (witdh * height)/2; i++){
		int yuyvIdx = i * 4;
		int rgbIdx = (i * 2) * 3;
		float y1 =(float) imgyuyv[yuyvIdx + 0];
		float u  =(float) imgyuyv[yuyvIdx + 1] - 128;
		float y2 =(float) imgyuyv[yuyvIdx + 2];
		float v  =(float) imgyuyv[yuyvIdx + 3] - 128;
		
		float inter1 = (1.402f * v);
		float inter2 = (0.344f * u);
		float inter3 = (0.714f * v);
		float inter4 = (1.772f * u);
		float rgb[3];

		rgb[0] = (y1 + inter1);
		rgb[1] = (y1 - inter2 - inter3);
		rgb[2] = (y1 + inter4);
		
		if (rgb[0] < 0){ rgb[0] = 0; } 
		if (rgb[1] < 0){ rgb[1] = 0; } 
		if (rgb[2] < 0){ rgb[2] = 0; }
		if (rgb[0] > 255 ){ rgb[0] = 255; } 
		if (rgb[1] > 255) { rgb[1] = 255; } 
		if (rgb[2] > 255) { rgb[2] = 255; }

		imgrgb[rgbIdx + 0] = (uint8_t)rgb[0]; 		      	
		imgrgb[rgbIdx + 1] = (uint8_t)rgb[1];   	      	
		imgrgb[rgbIdx + 2] = (uint8_t)rgb[2];   	      	

		rgb[0] = (y2 + inter1);
		rgb[1] = (y2 - inter2 - inter3);
		rgb[2] = (y2 + inter4);
		
		if (rgb[0] < 0){ rgb[0] = 0; } 
		if (rgb[1] < 0){ rgb[1] = 0; } 
		if (rgb[2] < 0){ rgb[2] = 0; }
		if (rgb[0] > 255 ){ rgb[0] = 255; } 
		if (rgb[1] > 255) { rgb[1] = 255; } 
		if (rgb[2] > 255) { rgb[2] = 255; }

		imgrgb[rgbIdx + 3] = (uint8_t)rgb[0]; 		      	
		imgrgb[rgbIdx + 4] = (uint8_t)rgb[1];   	      	
		imgrgb[rgbIdx + 5] = (uint8_t)rgb[2];
	}
}

bool set_framerate(int fd,int framerate){

    struct v4l2_streamparm parm;

    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = framerate;

    int ret = xioctl(fd, VIDIOC_S_PARM, &parm);

    if (ret < 0)
    {
        return false;
    }

    return true;
}


int xioctl(int fd, int request, void *arg)
{
    int r;
	do r = ioctl (fd, request, arg);
	while (-1 == r && EINTR == errno);
	return r;
}
