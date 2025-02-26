#!/usr/bin/env python

import sys
import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

sys.path.append("..")
from STservo_sdk import *

# Default setting
STS_ID_AZIMUTH             = 1                 # Azimuth servo ID
STS_ID_ELEVATION           = 2                 # Elevation servo ID
BAUDRATE                   = 1000000           # STServo default baudrate
DEVICENAME                 = 'COM6'            # Serial port
STS_MOVING_SPEED          = 2000              # STServo moving speed
STS_MOVING_ACC            = 50               # STServo moving acc
STS_MINIMUM_POSITION      = 0                 # Minimum position value
STS_MAXIMUM_POSITION      = 4095              # Maximum position value

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = sts(portHandler)

def initialize_port():
    # Open port
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        return False

    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        return False
    
    return True

def move_servo(id, position):
    # Write servo position with default speed and acceleration
    sts_comm_result, sts_error = packetHandler.WritePosEx(id, position, STS_MOVING_SPEED, STS_MOVING_ACC)
    
    if sts_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(sts_comm_result))
        return False
    elif sts_error != 0:
        print("%s" % packetHandler.getRxPacketError(sts_error))
        return False
    
    return True

def wait_for_servo(id):
    while True:
        # Read servo position and speed
        position, speed, comm_result, error = packetHandler.ReadPosSpeed(id)
        if comm_result == COMM_SUCCESS:
            print(f"[ID:{id:03d}] Position:{position} Speed:{speed}")
        
        # Check if servo is still moving
        moving, comm_result, error = packetHandler.ReadMoving(id)
        if comm_result == COMM_SUCCESS and moving == 0:
            break

def main():
    if not initialize_port():
        return

    print("\nServo Turret Control")
    print("-------------------")
    print("Enter angles between 0 and 4095 for both servos")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            try:
                # Get user input for both angles
                azimuth = int(input("\nEnter azimuth angle (0-4095): "))
                elevation = int(input("Enter elevation angle (0-4095): "))
                
                # Validate input
                if not (0 <= azimuth <= 4095 and 0 <= elevation <= 4095):
                    print("Error: Angles must be between 0 and 4095")
                    continue
                
                # Move both servos
                print("\nMoving to position...")
                if move_servo(STS_ID_AZIMUTH, azimuth) and move_servo(STS_ID_ELEVATION, elevation):
                    # Wait for both servos to complete their movement
                    wait_for_servo(STS_ID_AZIMUTH)
                    wait_for_servo(STS_ID_ELEVATION)
                    print("\nMovement completed!")
                
            except ValueError:
                print("Error: Please enter valid numbers")
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        # Close port
        portHandler.closePort()
        print("Port closed")

if __name__ == "__main__":
    main()