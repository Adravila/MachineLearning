# Machine Learning - Aprendizaje supervisado
# Por Adrian Davila Guerra

import numpy as np
from sklearn import tree

# HP, Mana, DPS, Heal
atributos = [ 
    [20,50,1,250],       # Healer - Sacerdote
    [23,90,4,170] ,      # Healer - Paladin
    [50,5,6,12],         # Tanque - DK 
    [61,0,14,0],         # Tanque - Guerrero
    [21,30,50,0],        # DPS - Mago 
    [21,12,44,12],       # DPS - Druida 
    [26,0,60,0],         # DPS - Picaro 
    [23,40,71,0],        # DPS - Mago 
    [23,20,65,12],       # DPS - Paladin 
    [21,20,66,35]]       # DPS - Monje
etiquetas = [0,0,1,1,2,2,2,2,2,2] # 0: Healer, 1: Tank, 2: DPS

clasificador = tree.DecisionTreeClassifier()
train = clasificador.fit(atributos,etiquetas)

print ("Introduzca la salud: ", end="") 
hp = input()
print ("Introduzca el mana: ", end="") 
mana = input()
print ("Introduzca el DPS: ", end="")
dps = input()
print ("Introduzca la cantidad de salud curada: ", end="") 
heal = input()

res = train.predict([[hp,mana,dps,heal]])

def switch(var):
    switcher = {
        0: "Healer", 
        1: "Tank", 
        2: "DPS",
    }
    return switcher.get(var,"Invalid rol")

print ("El rol que determina los datos introducidos es:", switch(int(res)))
