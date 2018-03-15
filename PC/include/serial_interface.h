
#ifndef _SERIAL_INTERFACE_H_
#define _SERIAL_INTERFACE_H_

void
set_blocking (int fd, int should_block);

int
set_interface_attribs (int fd, int speed, int parity);

int
open_com_port(const char* portname);


void close_com_port(int fd);

int write_bytes(int fd, char *buffer, int size);

void write_byte(int fd, char byte);

int read_bytes(int fd, char *buffer, int bytes_to_read);

char readbyte(int fd);


#endif
