#c compiler
CC = avr-gcc

CFLAGS = -Wall -g -Os -mmcu=atmega328p
MCU=atmega328p
F_CPU=1600000
CC=avr-gcc
OBJCOPY=avr-objcopy
CFLAGS=-std=c99 -Wall -g -Os -mmcu=${MCU} -DF_CPU=${F_CPU} -I.
TARGET=main
SRCS=main.c

all:
		${CC} ${CFLAGS} -o ${TARGET}.bin ${SRCS}
		${OBJCOPY} -j .text -j .data -O ihex ${TARGET}.bin ${TARGET}.hex

flash:
		avrdude -p ${MCU} -c arduino -U flash:w:${TARGET}.hex:i -F -P /dev/ttyUSB0 -b57600

clean:
		rm -f *.bin *.hex
