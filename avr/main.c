
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdint.h>

#define CCW 0
#define CW 1

uint8_t motor_speed = 100;
int8_t motor_dir = CCW;
uint8_t servo_pos = 95;

uint16_t ms = 0;

//will be true if the next read is the motor speed else false
uint8_t rx_toggle = 1;
uint8_t data_available = 0;
uint8_t ms_elapsed = 0;


//RX complete interrupt
ISR(USART_RX_vect){
	data_available = 1;
}

ISR(TIMER1_COMPA_vect){
	ms++;
	ms_elapsed = 1;
}

void 
init(){
	DDRD |= (1 << PIND1) |
			(1 << PIND3) |		        //servo
			(1 << PIND4) |			//enable
			(1 << PIND5) | (1 << PIND6);	//motor dir

	//motor control
	
	TCCR0A |= (1 << WGM00)  |				//Phase correct PWM mode.
			  (1 << COM0A1) |				//Clear OC0A on Compare Match when up-counting.
											//Set OC0A on Compare Match when down-counting.
			  (1 << COM0B1);				//Clear OC0B on Compare Match when up-counting.
											//Set OC0B on Compare Match when down-counting.
	
	TCCR0B |= (1 << CS01);					//Set the timer prescaler to 8 this will give PWM freq of 3906Hz.

	//servo control
	
	TCCR2A |= (1 << WGM20)  |
			  (1 << COM2A1);				//Clear OC2A on Compare Match when up-counting.
											//Set OC2A on Compare Match when down-counting.
	TCCR2B |= (1 << CS20);					//Set the timer prescaler to 1

	//com port
	
	UCSR0B |= (1 << RXCIE0) |				//Enable RX complete interrupt
			  (1 << TXEN0)	|				//Enable TX
			  (1 << RXEN0);					//Enable RX
	
	UCSR0C |= (1 << UCSZ00) |		
			  (1 << UCSZ01);				//Set word size to 8bit
	
	int UBBRValue = 12;						//UBBR of 12 will give me a baud rate of 76800 at 16MHz
	UBRR0H = (unsigned char) (UBBRValue >> 8);
	UBRR0L = (unsigned char) (UBBRValue);

	//timing
	
	TCCR1A |= (1 << CS10);					//timer prescaler to 1
	
	TCCR1B |= (1 << WGM13);					//CSC mode

	OCR1AL = 0b00000001;					//Set A Compare to 15873 so that the timer 
											//interrupt will be triggered every 1ms.
	OCR1AH = 0b00111110;

	TIMSK1 |= (1 << OCIE0A);					//Output Compare A Match Interrupt Enable

}		


int 
main(){
	init();
	
	
	//global interrupt enable
	sei();

	start_motor();
	while(1){
		if(data_available == 1){
			int8_t val = UDR0;
			if(rx_toggle){
				if(val > 0){
					motor_speed = val * 2;
					motor_dir = CCW;
				}else{
					motor_speed = val * -2;
					motor_dir = CW;
				}
			}else{
				if(servo_pos - val < 10)
				{
					servo_pos = 10;
				}else{
					servo_pos -= val;
				}
			}
			if(rx_toggle == 1){
				rx_toggle = 0;
			}else{
			 	rx_toggle = 1;
			}
			data_available = 0;
		}
		if(ms > 100){
			motor_speed = 0;
			while ( !( UCSR0A & (1<<UDRE0)));
			UDR0 = motor_speed;	
		}

	}
}

void stop_motor(){
	PORTD &= ~(1 << PIND4);
}

void start_motor(){
	PORTD |= (1 << PIND4);
}

void set_motor_speed(){
	if(motor_dir == CCW){
		OCR0A = 0;
		OCR0B = motor_speed;
	}else if(motor_dir == CW){
		OCR0A = motor_speed;
		OCR0B = 0;
	}
}


