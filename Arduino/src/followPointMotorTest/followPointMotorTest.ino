
#include <Servo.h>

#define DIR_CW 0
#define DIR_CCW 1

#define PIN_IN_1 7
#define PIN_E 2

#define PIN_SERVO 9 

Servo myservo;

int servopos = 95;

int motorontime = 0;

void setup(){
  
  myservo.attach(PIN_SERVO);
  myservo.write(servopos);
  
  Serial.begin(115200);
  
  pinMode(PIN_IN_1, OUTPUT);
  //pinMode(PIN_E, OUTPUT);
  
  set_rotation_dir(DIR_CW);
  start_motor();
  delay(20);
  soft_stop();
  
  delay(100);
  
  set_rotation_dir(DIR_CW);
  start_motor();
  delay(20);
  soft_stop();
}

void loop(){
  
  set_rotation_dir(DIR_CCW);
  start_motor();
  delay(200);
  soft_stop();
  
  servopos += 30;
  myservo.write(servopos);
  
  delay(1000);
  
  set_rotation_dir(DIR_CW);
  start_motor();
  delay(200);
  soft_stop();
  
  servopos -= 30;
  myservo.write(servopos);
  delay(1000);
}

void set_rotation_dir(int dir){
  if(DIR_CW == dir)
  {
    digitalWrite(PIN_IN_1, LOW);
  }else{ 
    digitalWrite(PIN_IN_1, HIGH);
  }
}

void start_motor(){
  digitalWrite(PIN_E, HIGH);

}

void soft_stop(){
  digitalWrite(PIN_E, LOW);

}

