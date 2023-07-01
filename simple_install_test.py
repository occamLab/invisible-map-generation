import g2o

v = g2o.VertexSE3()
print(v.estimate().translation())
