String serialString = "";
String serialString2 = "none";

void setup() {

  Serial.begin(115200);
  
}

void loop() {

  if (Serial.available()) {
    serialString = Serial.readStringUntil('#');
    if (Serial.available()) {
      serialString2 = Serial.readStringUntil('#');
    }
    Serial.print(serialString2);
    Serial.print(",");
    Serial.print(serialString);
    Serial.print("*");
  }
}

