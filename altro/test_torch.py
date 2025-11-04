import torch
x = torch.rand( 5, 3 )
print( x )
print( "E' presente la GPU?: " , torch.cuda.is_available() )