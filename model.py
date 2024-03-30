import torch
import torch.nn as nn
import torch.nn.functional as F

debug = False

# Model for Session 6 assignment

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3) #input -? OUtput? RF
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dp1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dp2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(8, 24, 3)
        self.bn3 = nn.BatchNorm2d(24)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dp3 = nn.Dropout(0.25)
        self.conv4 = nn.Conv2d(24, 48, 3)
        self.bn4 = nn.BatchNorm2d(48)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dp4 = nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(48, 10, 3)
        self.bn5 = nn.BatchNorm2d(10)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.dp5 = nn.Dropout(0.25)
        self.gap = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        # print(f"Start: {x.shape}")
        x = self.bn1(F.relu(self.conv1(x)))
        # print(f"Conv1: {x.shape}")
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        # print(f"Conv2: {x.shape}")
        x = self.dp3(self.bn3(F.relu(self.conv3(x))))
        # print(f"Conv3: {x.shape}")
        x = self.pool4(self.bn4(F.relu(self.conv4(x))))
        # print(f"Conv4: {x.shape}")
        x = self.dp5(self.bn5(F.relu(self.conv5(x))))
        # print(f"Conv5: {x.shape}")
        x = self.gap(x)
        # print(f"GAP: {x.shape}")
        # x = torch.flatten(x, start_dim=1)
        x = x.view(-1, 10)
        # print(f"Flatten: {x.shape}")
        # x = self.fc1(x)
        # print(f"FC: {x.shape}")
        return F.log_softmax(x, dim=1)


# Session 7 Model for Step 1
class Net7_1(nn.Module):
    def __init__(self):
        super(Net7_1, self).__init__()
        # Input Block
        self.convblock3_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock3_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock1_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock3_5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 9
        self.convblock3_6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock1_7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        self.fc1 = nn.Linear(1568, 100, bias=False) # 7 * 7 * 32
        self.fc2 = nn.Linear(100, 10, bias=False)

    def forward(self, x):
        if (debug == True):
          print(f"Start: {x.shape}")
        x = self.convblock3_1(x)  # channel size 28 > 26  RF 1 > 3    J 1 > 1
        if (debug == True):
          print(f"Conv3_1: {x.shape}")
        x = self.convblock3_2(x)  # channel size 26 > 24  RF 3 > 5    J 1 > 1
        if (debug == True):
          print(f"Conv3_2: {x.shape}")
        x = self.convblock3_3(x)  # channel size 24 > 22  RF 5 > 7    J 1 > 1
        if (debug == True):
          print(f"Conv3_3: {x.shape}")
        x = self.pool1(x)         # channel size 22 > 11  RF 7 > 8    J 1 > 2
        if (debug == True):
          print(f"Pool1: {x.shape}")
        x = self.convblock1_4(x)  # channel size 11 > 11  RF 8 > 8    J 2 > 2
        if (debug == True):
          print(f"Conv1_4: {x.shape}")
        x = self.convblock3_5(x)  # channel size 11 > 9   RF 8 > 12   J 2 > 2
        if (debug == True):
          print(f"Conv3_5: {x.shape}")
        x = self.convblock3_6(x)  # channel size 9 > 7    RF 12 > 16  J 2 > 2
        if (debug == True):
          print(f"Conv3_6: {x.shape}")
        x = self.convblock1_7(x)  # channel size 7 > 7    RF 16 > 16  J 2 > 2
        if (debug == True):
          print(f"Conv1_7: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        # x = x.view(-1, 1568)        # 7 * 7 * 32
        if (debug == True):
          print(f"Flatten: {x.shape}")
        x = self.fc1(x)
        if (debug == True):
          print(f"FC1: {x.shape}")
        x = self.fc2(x)
        if (debug == True):
          print(f"FC2: {x.shape}")
        return F.log_softmax(x, dim=-1)
        
# Session 7 - Model for Step 2
class Net7_2(nn.Module):
    def __init__(self):
        super(Net7_2, self).__init__()
        # Input Block
        self.convblock3_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock3_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3_3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock1_4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock3_5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 9
        self.convblock3_6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock1_7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        self.fc1 = nn.Linear(490, 10, bias=False) # 7 * 7 * 10

    def forward(self, x):
        if (debug == True):
          print(f"Start: {x.shape}")
        x = self.convblock3_1(x)  # channel size 28 > 26  RF 1 > 3    J 1 > 1
        if (debug == True):
          print(f"Conv3_1: {x.shape}")
        x = self.convblock3_2(x)  # channel size 26 > 24  RF 3 > 5    J 1 > 1
        if (debug == True):
          print(f"Conv3_2: {x.shape}")
        x = self.convblock3_3(x)  # channel size 24 > 22  RF 5 > 7    J 1 > 1
        if (debug == True):
          print(f"Conv3_3: {x.shape}")
        x = self.pool1(x)         # channel size 22 > 11  RF 7 > 8    J 1 > 2
        if (debug == True):
          print(f"Pool1: {x.shape}")
        x = self.convblock1_4(x)  # channel size 11 > 11  RF 8 > 8    J 2 > 2
        if (debug == True):
          print(f"Conv1_4: {x.shape}")
        x = self.convblock3_5(x)  # channel size 11 > 9   RF 8 > 12   J 2 > 2
        if (debug == True):
          print(f"Conv3_5: {x.shape}")
        x = self.convblock3_6(x)  # channel size 9 > 7    RF 12 > 16  J 2 > 2
        if (debug == True):
          print(f"Conv3_6: {x.shape}")
        x = self.convblock1_7(x)  # channel size 7 > 7    RF 16 > 16  J 2 > 2
        if (debug == True):
          print(f"Conv1_7: {x.shape}")
        x = x.view(-1, 490)        # 7 * 7 * 10
        if (debug == True):
          print(f"Flatten: {x.shape}")
        x = self.fc1(x)
        if (debug == True):
          print(f"FC1: {x.shape}")
        return F.log_softmax(x, dim=-1)        
        
# Session 7 - Model for Step 3
class Net7_3(nn.Module):
    def __init__(self):
        super(Net7_3, self).__init__()
        # Input Block
        self.convblock3_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock3_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3_3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock1_4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock3_5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 9
        self.convblock3_6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock1_7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1

        self.fc1 = nn.Linear(490, 10, bias=False) # 7 * 7 * 10
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        if (debug == True):
          print(f"Start: {x.shape}")
        x = self.convblock3_1(x)  # channel size 28 > 26  RF 1 > 3    J 1 > 1
        if (debug == True):
          print(f"Conv3_1: {x.shape}")
        x = self.dropout(x)
        x = self.convblock3_2(x)  # channel size 26 > 24  RF 3 > 5    J 1 > 1
        if (debug == True):
          print(f"Conv3_2: {x.shape}")
        x = self.dropout(x)
        x = self.convblock3_3(x)  # channel size 24 > 22  RF 5 > 7    J 1 > 1
        if (debug == True):
          print(f"Conv3_3: {x.shape}")
        x = self.dropout(x)
        x = self.pool1(x)         # channel size 22 > 11  RF 7 > 8    J 1 > 2
        if (debug == True):
          print(f"Pool1: {x.shape}")
        x = self.convblock1_4(x)  # channel size 11 > 11  RF 8 > 8    J 2 > 2
        if (debug == True):
          print(f"Conv1_4: {x.shape}")
        x = self.dropout(x)
        x = self.convblock3_5(x)  # channel size 11 > 9   RF 8 > 12   J 2 > 2
        if (debug == True):
          print(f"Conv3_5: {x.shape}")
        x = self.dropout(x)
        x = self.convblock3_6(x)  # channel size 9 > 7    RF 12 > 16  J 2 > 2
        if (debug == True):
          print(f"Conv3_6: {x.shape}")
        x = self.dropout(x)
        x = self.convblock1_7(x)  # channel size 7 > 7    RF 16 > 16  J 2 > 2
        if (debug == True):
          print(f"Conv1_7: {x.shape}")
        x = self.dropout(x)
        x = self.gap(x)
        if (debug == True):
          print(f"GAP: {x.shape}")
        x = x.view(-1, 10)        # 7 * 7 * 10
        if (debug == True):
          print(f"Flatten: {x.shape}")
        # x = self.fc1(x)
        # if (debug == True):
        #   print(f"FC1: {x.shape}")
        return F.log_softmax(x, dim=-1)        
        
# Session 7 - Model for Step 4
dropout_value = 0.1
class Net7_4(nn.Module):
    def __init__(self):
        super(Net7_4, self).__init__()
        # Input Block
        self.convblock3_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock3_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12
        self.convblock3_3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 10

        # TRANSITION BLOCK 1
        self.convblock1_4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 10

        # CONVOLUTION BLOCK 2
        self.convblock3_5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock3_6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock3_6_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 4

        # OUTPUT BLOCK
        self.convblock1_7 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 4

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1

    def forward(self, x):
        if (debug == True):
          print(f"Start: {x.shape}")
        x = self.convblock3_1(x)  # channel size 28 > 26  RF 1 > 3    J 1 > 1
        if (debug == True):
          print(f"Conv3_1: {x.shape}")
        x = self.convblock3_2(x)  # channel size 26 > 24  RF 3 > 5    J 1 > 1
        if (debug == True):
          print(f"Conv3_2: {x.shape}")
        x = self.pool1(x)         # channel size 24 > 12  RF 5 > 6    J 1 > 2
        if (debug == True):
          print(f"Pool1: {x.shape}")
        x = self.convblock3_3(x)  # channel size 12 > 10  RF 6 > 10    J 2 > 2
        if (debug == True):
          print(f"Conv3_3: {x.shape}")
        x = self.convblock1_4(x)  # channel size 10 > 10  RF 10 > 14    J 2 > 2
        if (debug == True):
          print(f"Conv1_4: {x.shape}")
        x = self.convblock3_5(x)  # channel size 10 > 8   RF 14 > 18   J 2 > 2
        if (debug == True):
          print(f"Conv3_5: {x.shape}")
        x = self.convblock3_6(x)  # channel size 8 > 6    RF 18 > 22  J 2 > 2
        if (debug == True):
          print(f"Conv3_6: {x.shape}")
        x = self.convblock3_6_1(x)  # channel size 6 > 4    RF 22 > 26  J 2 > 2
        if (debug == True):
          print(f"Conv3_6_1: {x.shape}")
        x = self.convblock1_7(x)  # channel size 4 > 4    RF 26 > 26  J 2 > 2
        if (debug == True):
          print(f"Conv1_7: {x.shape}")
        x = self.gap(x)
        if (debug == True):
          print(f"GAP: {x.shape}")
        x = x.view(-1, 10)        # 4 * 4 * 10
        if (debug == True):
          print(f"Flatten: {x.shape}")
        # x = self.fc1(x)
        # if (debug == True):
        #   print(f"FC1: {x.shape}")
        return F.log_softmax(x, dim=-1)
        
# Session 7 - Model for Step 5
dropout_value = 0.02
class Net7_5(nn.Module):
    def __init__(self):
        super(Net7_5, self).__init__()
        # Input Block
        self.convblock0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 28  RF = 3

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14  RF = 4
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 12 RF = 8
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 10 RF = 12
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 8 RF = 16
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 6 RF = 20
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 4 RF = 24
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 2 RF = 28
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 2 RF = 28

        # TRANSITION BLOCK 1
        self.convblock_t_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1

    def forward(self, x):
        if (debug == True):
          print(f"Start: {x.shape}")

        x = self.convblock0(x)  # channel size 28 > 28  RF 1 > 3    J 1 > 1
        if (debug == True):
          print(f"Conv0: {x.shape}")
        x = self.pool1(x)         # channel size 28 > 14  RF 3 > 4    J 1 > 2

        x = self.convblock1(x)  # channel size 14 > 12  RF 4 > 8    J 2 > 2
        if (debug == True):
          print(f"Conv1: {x.shape}")
        if (debug == True):
          print(f"Pool1: {x.shape}")
        x = self.convblock2(x)  # channel size 12 > 10  RF 8 > 12    J 2 > 2
        if (debug == True):
          print(f"Conv2: {x.shape}")
        x = self.convblock3(x)  # channel size 10 > 8  RF 12 > 16    J 2 > 2
        if (debug == True):
          print(f"Conv3: {x.shape}")
        x = self.convblock4(x)  # channel size 8 > 6  RF 16 > 20    J 2 > 2
        if (debug == True):
          print(f"Conv4: {x.shape}")
        x = self.convblock5(x)  # channel size 6 > 4  RF 20 > 24    J 2 > 2
        if (debug == True):
          print(f"Conv5: {x.shape}")
        x = self.convblock6(x)  # channel size 4 > 2  RF 24 > 28    J 2 > 2
        if (debug == True):
          print(f"Conv6: {x.shape}")

        x = self.gap(x)         # channel size 2 > 1  RF 28 > 28    J 2 > 2
        if (debug == True):
          print(f"GAP: {x.shape}")

        x = self.convblock_t_2(x)  # channel size 1 > 1  RF 28 > 28    J 2 > 2
        if (debug == True):
          print(f"Conv_t_2: {x.shape}")

        x = x.view(-1, 10)
        if (debug == True):
          print(f"Flatten: {x.shape}")

        return F.log_softmax(x, dim=-1)       
        
# Session 8 - Model with Batch Normalization
class Net8_Batch(nn.Module):
    def __init__(self, dropout_value):
        super(Net8_Batch, self).__init__()
        # Input Block
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 32

        # CONVOLUTION BLOCK 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        # TRANSITION BLOCK 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 30

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 28
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 11

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 13

        # TRANSITION BLOCK 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 11

        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 9
        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 7
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 5

        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=6) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1

        # TRANSITION BLOCK 1
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

    def forward(self, x, debug=False):
        if (debug == True):
          print(f"Start: {x.shape}")


        x = self.C1(x)  # channel size 32 > 32  RF 1 > 3    J 1 > 1
        if (debug == True):
          print(f"C1: {x.shape}")
        x = self.C2(x)  # channel size 32 > 30  RF 3 > 5    J 1 > 1
        if (debug == True):
          print(f"C2: {x.shape}")
        x = self.c3(x)  # channel size 30 > 30  RF 5 > 5    J 1 > 1
        if (debug == True):
          print(f"c3: {x.shape}")
        x = self.C4(x)  # channel size 30 > 28  RF 5 > 7    J 1 > 1
        if (debug == True):
          print(f"C4: {x.shape}")
        x = self.C5(x)  # channel size 28 > 26  RF 7 > 9    J 1 > 1
        if (debug == True):
          print(f"C5: {x.shape}")
        x = self.pool1(x) # channel size 26 > 13  RF 9 > 10    J 1 > 2
        if (debug == True):
          print(f"Pool1: {x.shape}")
        x = self.C6(x)  # channel size 13 > 11  RF 10 > 14    J 2 > 2
        if (debug == True):
          print(f"C6: {x.shape}")
        x = self.c7(x)  # channel size 11 > 11  RF 14 > 18    J 2 > 2
        if (debug == True):
          print(f"c7: {x.shape}")
        x = self.C8(x)  # channel size 11 > 9  RF 18 > 18    J 2 > 2
        if (debug == True):
          print(f"C8: {x.shape}")
        x = self.C9(x)  # channel size 9 > 7  RF 18 > 22    J 2 > 2
        if (debug == True):
          print(f"C9: {x.shape}")
        x = self.C10(x) # channel size 7 > 5  RF 22 > 26    J 2 > 2
        if (debug == True):
          print(f"C10: {x.shape}")

        x = self.GAP(x) # channel size 5 > 1
        if (debug == True):
          print(f"GAP: {x.shape}")

        x = self.c11(x)  # channel size 1 > 1  RF 26 > 26    J 2 > 2
        if (debug == True):
          print(f"c11: {x.shape}")

        x = x.view(-1, 10)
        if (debug == True):
          print(f"Flatten: {x.shape}")

        return F.log_softmax(x, dim=-1)
        
# Session 8 - Model with Layer Normalization
class Net8_Layer(nn.Module):
    def __init__(self, dropout_value = 0.1):
        super(Net8_Layer, self).__init__()
        # Input Block
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([32,32]),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 32  RF = 3

        # CONVOLUTION BLOCK 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm([30, 30]),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 30 RF = 8

        # TRANSITION BLOCK 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 30


        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm([28, 28]),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 28
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm([26, 26]),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 13

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm([11, 11]),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 11

        # TRANSITION BLOCK 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 11

        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm([9, 9]),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 9
        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm([7, 7]),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 7
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm([5, 5]),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 5

        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=5) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1

        # TRANSITION BLOCK 1
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

    def forward(self, x):
        if (debug == True):
          print(f"Start: {x.shape}")

        x = self.C1(x)  # channel size 32 > 32  RF 1 > 3    J 1 > 1
        if (debug == True):
          print(f"C1: {x.shape}")
        x = self.C2(x)  # channel size 32 > 30  RF 3 > 5    J 1 > 1
        if (debug == True):
          print(f"C2: {x.shape}")
        x = self.c3(x)  # channel size 30 > 30  RF 5 > 5    J 1 > 1
        if (debug == True):
          print(f"c3: {x.shape}")
        x = self.C4(x)  # channel size 30 > 28  RF 5 > 7    J 1 > 1
        if (debug == True):
          print(f"C4: {x.shape}")
        x = self.C5(x)  # channel size 28 > 26  RF 7 > 9    J 1 > 1
        if (debug == True):
          print(f"C5: {x.shape}")
        x = self.pool1(x) # channel size 16 > 13  RF 9 > 10    J 1 > 2
        if (debug == True):
          print(f"Pool1: {x.shape}")
        x = self.C6(x)  # channel size 13 > 11  RF 10 > 14    J 2 > 2
        if (debug == True):
          print(f"C6: {x.shape}")
        x = self.c7(x)  # channel size 11 > 11  RF 14 > 14    J 2 > 2
        if (debug == True):
          print(f"c7: {x.shape}")
        x = self.C8(x)  # channel size 11 > 9  RF 14 > 18    J 2 > 2
        if (debug == True):
          print(f"C8: {x.shape}")
        x = self.C9(x)  # channel size 9 > 7  RF 18 > 22    J 2 > 2
        if (debug == True):
          print(f"C9: {x.shape}")
        x = self.C10(x) # channel size 7 > 5  RF 22 > 26    J 2 > 2
        if (debug == True):
          print(f"C10: {x.shape}")

        x = self.GAP(x) # channel size 5 > 1
        if (debug == True):
          print(f"GAP: {x.shape}")

        x = self.c11(x)  # channel size 1 > 1  RF 28 > 28    J 2 > 2
        if (debug == True):
          print(f"c11: {x.shape}")

        x = x.view(-1, 10)
        if (debug == True):
          print(f"Flatten: {x.shape}")

        return F.log_softmax(x, dim=-1)
        
# Session 8 - Model with Group Normalization
class Net8_Group(nn.Module):
    def __init__(self, num_groups=8, dropout_value = 0.1):
        super(Net8_Group, self).__init__()
        # Input Block
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(num_groups, 16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 32  RF = 3

        # CONVOLUTION BLOCK 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 30 RF = 8

        # TRANSITION BLOCK 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 30


        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(num_groups, 16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 28
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 13

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 11

        # TRANSITION BLOCK 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 11

        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 9
        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 7
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 5

        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=5) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1

        # TRANSITION BLOCK 1
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

    def forward(self, x):
        if (debug == True):
          print(f"Start: {x.shape}")

        x = self.C1(x)  # channel size 32 > 32  RF 1 > 3    J 1 > 1
        if (debug == True):
          print(f"C1: {x.shape}")
        x = self.C2(x)  # channel size 32 > 30  RF 3 > 5    J 1 > 1
        if (debug == True):
          print(f"C2: {x.shape}")
        x = self.c3(x)  # channel size 30 > 30  RF 5 > 5    J 1 > 1
        if (debug == True):
          print(f"c3: {x.shape}")
        x = self.C4(x)  # channel size 30 > 28  RF 5 > 7    J 1 > 1
        if (debug == True):
          print(f"C4: {x.shape}")
        x = self.C5(x)  # channel size 28 > 26  RF 7 > 9    J 1 > 1
        if (debug == True):
          print(f"C5: {x.shape}")
        x = self.pool1(x) # channel size 16 > 13  RF 9 > 10    J 1 > 2
        if (debug == True):
          print(f"Pool1: {x.shape}")
        x = self.C6(x)  # channel size 13 > 11  RF 10 > 14    J 2 > 2
        if (debug == True):
          print(f"C6: {x.shape}")
        x = self.c7(x)  # channel size 11 > 11  RF 14 > 14    J 2 > 2
        if (debug == True):
          print(f"c7: {x.shape}")
        x = self.C8(x)  # channel size 11 > 9  RF 14 > 18    J 2 > 2
        if (debug == True):
          print(f"C8: {x.shape}")
        x = self.C9(x)  # channel size 9 > 7  RF 18 > 22    J 2 > 2
        if (debug == True):
          print(f"C9: {x.shape}")
        x = self.C10(x) # channel size 7 > 5  RF 22 > 26    J 2 > 2
        if (debug == True):
          print(f"C10: {x.shape}")

        x = self.GAP(x) # channel size 5 > 1
        if (debug == True):
          print(f"GAP: {x.shape}")

        x = self.c11(x)  # channel size 1 > 1  RF 28 > 28    J 2 > 2
        if (debug == True):
          print(f"c11: {x.shape}")

        x = x.view(-1, 10)
        if (debug == True):
          print(f"Flatten: {x.shape}")

        return F.log_softmax(x, dim=-1)
        
# 1, 2 and 3 layers should only have dilation and stride of 2 - Try in C3 and then in C1 and C2
# 2, 3 and 4 layers should have depthwise separable convolutions - First in C4 then C3 and then C2
# GAP followed by FC or 1 x 1 layer in output
# Use Albumentations library

class ConvBlock(nn.Module):

    def __init__(self, inc, outc, dropout_value=0.1, pad=1, dilate=1, stride=1, depthwise=True, last_layer=False):
        super(ConvBlock, self).__init__()

        if (depthwise):
            self.CN1 = nn.Sequential(
                nn.Conv2d(in_channels=inc,out_channels=inc,kernel_size=3,padding=pad,groups=inc,bias=False),
                nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=1,bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            )
        else:
            self.CN1 = nn.Sequential(
                nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=3,padding=pad,bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            )

        if (depthwise):
            self.CN2 = nn.Sequential(
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,groups=outc,bias=False),
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=1,bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            )
        else:
            self.CN2 = nn.Sequential(
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            )

        if (depthwise):
            self.CN3 = nn.Sequential(
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,groups=outc,bias=False),
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=1,bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            )
        else:
            self.CN3 = nn.Sequential(
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            )

        if (depthwise):
            self.CN4 = nn.Sequential(
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,groups=outc,bias=False),
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=1,bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            )
        else:
            self.CN4 = nn.Sequential(
                nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            )

        if (last_layer):
            if (depthwise):
                self.CNL = nn.Sequential(
                    nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,stride=stride, padding=pad,groups=outc,bias=False),
                    nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=1,bias=False),
                )
            else:
                self.CNL = nn.Sequential(
                    nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,stride=stride, bias=False),
                )
        else:
          if (dilate > 1):
            if (depthwise):
              self.CNL = nn.Sequential(
                  nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,stride=stride, dilation=2,groups=outc,bias=False),
                  nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=1,bias=False),
                  nn.BatchNorm2d(outc),
                  nn.ReLU(),
                  nn.Dropout(dropout_value),
              )
            else:
              self.CNL = nn.Sequential(
                  nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,stride=stride, dilation=2,bias=False),
                  nn.BatchNorm2d(outc),
                  nn.ReLU(),
                  nn.Dropout(dropout_value),
              )
          else:
            if (depthwise):
              self.CNL = nn.Sequential(
                  nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,padding=pad,stride=stride,groups=outc,bias=False),
                  nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=1,bias=False),
                  nn.BatchNorm2d(outc),
                  nn.ReLU(),
                  nn.Dropout(dropout_value),
              )
            else:
              self.CNL = nn.Sequential(
                  nn.Conv2d(in_channels=outc,out_channels=outc,kernel_size=3,stride=stride, padding=pad,bias=False),
                  nn.BatchNorm2d(outc),
                  nn.ReLU(),
                  nn.Dropout(dropout_value),
              )

    def forward(self, x):

        x = self.CN1(x)
        x = self.CN2(x)
        x = self.CN3(x)
        x = self.CN4(x)
        x = self.CNL(x)

        return x

class S9(nn.Module):
    def __init__(self, dropout_value):
        super(S9,self).__init__()

        self.InBlock = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )   # RF = 3    Output = 32

        self.C1 = ConvBlock(8, 16, dropout_value=dropout_value, pad=1, stride=1, dilate=2, depthwise=False,  last_layer=False)  # RF = 15   Output = 30
        self.C2 = ConvBlock(16, 32, dropout_value=dropout_value, pad=1, stride=1, dilate=2, depthwise=True, last_layer=False)   # RF = 39   Output = 28
        self.C3 = ConvBlock(32, 64, dropout_value=dropout_value, pad=2, stride=2, dilate=2, depthwise=True, last_layer=False)   # RF = 87   Output = 18
        self.CL = ConvBlock(64, 160, dropout_value=dropout_value, pad=0, stride=2, dilate=1, depthwise=True, last_layer=True)   # RF = 247  Output = 4

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # RF = 247

    def forward(self, x):

        x = self.InBlock(x)
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.CL(x)
        x = self.gap(x)
        x = self.conv11(x)
        x = x.view(x.shape[0], -1)

        return F.log_softmax(x, dim=-1)
