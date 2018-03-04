
#include <Servo.h>

#define DIR_CW 0
#define DIR_CCW 1

#define PIN_IN_1 9
#define PIN_IN_2 10
#define PIN_E 4

#define PIN_SERVO 5

#define MIN_SERVO_POS 50
#define MAX_SERVO_POS 200


Servo myservo;
byte servopos = 95;

int motordir = DIR_CW;
int motorspeed = 100;

void setup(){
  
  myservo.attach(PIN_SERVO);
  Serial.begin(115200);
  
  pinMode(PIN_IN_1, OUTPUT);
  pinMode(PIN_IN_2, OUTPUT);
  //pinMode(PIN_E, OUTPUT);
  
  start_motor();
}

void loop(){
  if(servopos > MIN_SERVO_POS && servopos < MAX_SERVO_POS){
      myservo.write(servopos);
  }
  if(motorspeed < 20){
    stop_motor();
  }else{
    start_motor();
  }
  setmotorspeed();
  motorspeed--;
  delay(1);
}

void serialEvent() {
  while (Serial.available() > 1) {
  
    int8_t dx = (int8_t)Serial.read();
    int8_t dy = (int8_t)Serial.read();
    
    servopos -= dy;
    
    if(dx > 0){
      motordir = 0;
      motorspeed = dx;
    }else if(dx < 0){
      motordir = 1;
      motorspeed = dx * -1;
    }else{
      motorspeed = 0;  
    }
  }
}

void setmotorspeed(){
  if(DIR_CW == motordir)
  {
    digitalWrite(PIN_IN_1, HIGH);
    analogWrite(PIN_IN_2,motorspeed);
  }else{
    digitalWrite(PIN_IN_2, HIGH);
    analogWrite(PIN_IN_1,motorspeed);
  }
}

void start_motor(){
  digitalWrite(PIN_E, HIGH);
}

void stop_motor(){
  digitalWrite(PIN_E, LOW);
}
