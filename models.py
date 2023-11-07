import torch 
import torch.nn as nn
import torch.nn.functional as F 

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, invert):
        super().__init__() 
        self.invert = invert
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=drop_rate)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))  # same (not valid) convolutions are used, i.e., image size doesn't change
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
    def forward(self, x): 
        x = F.relu(self.conv1(x), inplace=False)  
        if self.invert:
            x = self.batch_norm(x)                    
            x = self.dropout(x)
        else:
            x = self.dropout(x)
            x = self.batch_norm(x)                        
        x = F.relu(self.conv2(x), inplace=False)  
        return x
    
    
class UNet(nn.Module):
    def __init__(self, num_classes, drop_rate, input_channels, invert, depth):
        super().__init__()
        self.depth = depth
        self.invert = invert
        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm2d(input_channels))
        self.batch_norms.append(nn.BatchNorm2d(64))
        self.dropout = nn.Dropout(p=drop_rate)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.up_convs = nn.ModuleList()
        self.double_convs = nn.ModuleList()
        self.double_convs.append(DoubleConv(input_channels, 64, drop_rate, self.invert))
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(1,1), stride=(1,1), padding=0) 
        for i in range(self.depth):
            self.batch_norms.append(nn.BatchNorm2d(2**(7+i)))
            self.double_convs.append(DoubleConv(2**(6+i), 2**(7+i), drop_rate, self.invert))  # double convs in encoder
            self.up_convs.append(
                nn.ConvTranspose2d(
                    in_channels=2**(6 + self.depth - i), 
                    out_channels=2**(5 + self.depth - i), 
                    kernel_size=(2,2), 
                    stride=(2,2)
                )
            )
        for i in range(self.depth):
            self.double_convs.append(
                DoubleConv(
                    2**(6 + self.depth - i), 
                    2**(5 + self.depth - i), 
                    drop_rate, 
                    self.invert
                )
            )  # double convs in decoder
        
    def forward(self, x):
        # encoder part
        x = self.batch_norms[0](x) 
        skip_connections = [] 
        for i in range(self.depth):    
            x = self.double_convs[i](x)
            skip_connections.append(x)
            x = self.pool(x)
            if self.invert:
                x = self.batch_norms[i + 1](x)
                x = self.dropout(x)
            else:
                x = self.dropout(x)
                x = self.batch_norms[i + 1](x)
                        
        x = self.double_convs[i + 1](x)   
        
        # decoder part
        for i in range(self.depth):    
            if self.invert:
                x = self.batch_norms[self.depth + 1 - i](x)  
                x = self.dropout(x)
            else:  
                x = self.dropout(x)
                x = self.batch_norms[self.depth + 1 - i](x)
            x = self.up_convs[i](x)
            x = torch.cat((skip_connections[self.depth - 1 - i], x), dim=1)   
            if self.invert:
                x = self.batch_norms[self.depth + 1 - i](x) 
                x = self.dropout(x)
            else: 
                x = self.dropout(x)
                x = self.batch_norms[self.depth + 1 - i](x)
            x = self.double_convs[self.depth + 1 + i](x)
        if self.invert:
            x = self.batch_norms[1](x)    
            x = self.dropout(x)
        else:    
            x = self.dropout(x)
            x = self.batch_norms[1](x)
        x = self.final_conv(x)
        return x