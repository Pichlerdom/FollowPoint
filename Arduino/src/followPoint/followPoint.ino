
#include <Servo.h>

#define DIR_CW 0
#define DIR_CCW 1

#define PIN_IN_1 2
#define PIN_E 5

#define PIN_SERVO 9 

#define MIN_SERVO_POS 10
#define MAX_SERVO_POS 250


Servo myservo;
byte servopos = 95;

int motordir = 0;
int motorontime = 3;

void setup(){
  
  myservo.attach(PIN_SERVO);
  Serial.begin(115200);
  
  pinMode(PIN_IN_1, OUTPUT);
  //pinMode(PIN_E, OUTPUT);
  
  startupmotion();

}

void loop(){
  if(servopos > MIN_SERVO_POS && servopos < MAX_SERVO_POS){
      myservo.write(servopos);
  }
  
  if(motorontime > 0){
    motorontime --;
   
    set_rotation_dir(motordir);
    start_motor();
  }else{
    soft_stop();
  }
  delay(1);
}

void serialEvent() {
  while (Serial.available() > 1) {
  
    int8_t dx = (int8_t)Serial.read();
    int8_t dy = (int8_t)Serial.read();
    
    servopos -= dy;
    
    if(dx > 0){
      motordir = 0;
      motorontime = dy;
    }else if(dx < 0){
      motordir = 1;
      motorontime = dy * -1;
    }else{
      motorontime = 0;  
    }
    Serial.write(dx);
    Serial.write(dy);
  }
}

void startupmotion(){
  
  myservo.write(servopos);
  
  set_rotation_dir(DIR_CW);
  start_motor();
  delay(50);
  soft_stop();
  
  delay(100);
  
  set_rotation_dir(DIR_CCW);
  start_motor();
  delay(50);
  soft_stop();
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
