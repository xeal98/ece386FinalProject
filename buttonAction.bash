#!/bin/bash

#https://www.datacamp.com/tutorial/how-to-write-bash-script-tutorial
#https://www.cyberciti.biz/faq/bash-while-loop/
#^For the while loop

gpioinfo | grep 105 #Pin #29 Input Active-High

gpioget gpiochip0 105

#while [ 1 -le 5 ]
#do
#    if [gpiomon --bias=pull-down -r gpiochip0 105 -eq 1];
#        then
#            echo "Rising Edge Detected"
#    fi
#done

gpiomon --bias=pull-down -r -n 1 gpiochip0 105 | while read line; do echo "event $line"; done
