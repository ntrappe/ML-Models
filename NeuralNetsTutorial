## Tutorial on what's happening in the code

### In the function 'backward':

Parameters:
```
self.inputSize = 4
self.hiddenSize = 3
self.outputSize = 1
n = 145
```

Calculate the error from the expected and actual output:
```
self.o_error = y - o
# (145,1) - (145,1) = (145,1)
```

Determine how much of an error it is by doing the cost function on it:
```
self.o_delta = self.o_error * sig_deriv(o)
# (145,1)
```

Calculate how much the hidden layer is contributing to the output error:
```
self.z2_error = self.o_delta.dot(self.weight_H.T)
# (145,1) * (1,3) = (145,3)
```

Take the cost function of the error to see how much it's contributing:
```
self.z2_delta = self.z2_error * sig_deriv(self.z2)
# (145,3) * (145,3) = (145,3)
```

Adjust the weights by seeing how much they contributed to error:
```
self.weight_I += X.T.dot(self.z2_delta)
# (4,145) * (145,3) = (4,3)
```
```
self.weight_H += self.z2.T.dot(self.o_delta)
# (3,145) * (145,1) = (3,1)
```

### For a DEEPER net, in the function 'backward':

Parameters:
```
self.inputSize = 30
self.hiddenSize = 10
self.hidden2Size = 5
self.outputSize = 1
n = 45
```

Calculate the error from the expected and actual output:
```
self.o_error = y - o
# (45,1) - (45,1) = (45,1)
```

Determine how much of an error it is by doing the cost function on it:
```
self.o_delta = self.o_error * sig_deriv(o)
# (45,1)
```

Calculate how much the second hidden layer is contributing to the output error:
```
self.z4_error = self.o_delta.dot(self.weight_H2.T)
# (45,1) * (1,5) = (45,5)
```

Take the cost function of the error to see how much it's contributing:
```
self.z4_delta = self.z4_error * sig_deriv(self.z4)
# (45,5) * (45,5) = (45,5)
```

Calculate how much the first hidden layer is contributing to the output error:
```
self.z2_error = self.z4_delta.dot(self.weight_H.T)
# (45,5) * (5,10) = (45,10)
```

Take the cost function of the error to see how much it's contributing:
```
self.z2_delta = self.z2_error * sig_deriv(self.z2)
```

Adjust the weights:
```
self.weight_H2 += self.z4.T.dot(self.o_delta)
# (5,45) * (45,1) = (5,1)
```
```
self.weight_H += self.z2.T.dot(self.z4_delta)
# (10,45) * (45,5) = (10,5)
```
```
self.weight_I += X.T.dot(self.z2_delta)
# (30,45) * (45,10) = (30,10)
```




