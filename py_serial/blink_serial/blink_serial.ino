void setup() 
{
  pinMode(13,OUTPUT);
  digitalWrite(13,LOW);  
  Serial.begin(9600);
}

void loop() 
{
  if(Serial.available() > 0)
  {
    int data=Serial.read();
    Serial.println(data);
    if(data == 49)
    {
      digitalWrite(13,HIGH);
      delay(1000);
    }
     else if(data == 48)
    {
      digitalWrite(13,HIGH);
      delay(5000);
    }
  }  
    else
    {
      digitalWrite(13,LOW);
    }
}
