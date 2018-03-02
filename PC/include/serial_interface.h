
#ifndef _SERIAL_INTERFACE_H_
#define _SERIAL_INTERFACE_H_


#include <stdint.h>

void
set_blocking (int fd, int should_block);

int
set_interface_attribs (int fd, int speed, int parity);

int
open_com_port(const char* portname);

void
write_delta(int fd, int8_t dx, int8_t dy);
#endif
