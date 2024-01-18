#include <Servo.h>

Servo myservo;

void setup() {
  myservo.attach(3);
  Serial.begin(9600);
}

void loop() {

  if(Serial.available()>0){
    int incomingByte = Serial.read();

    if(incomingByte=='1'){
      myservo.write(180);
    }
    else if(incomingByte=='2'){
      myservo.write(0);
    }
    else{
      myservo.write(90);
    }
  }

}
