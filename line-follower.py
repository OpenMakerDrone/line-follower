#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© Copyright 2016, Team EverywhereDrone at Yonsei University.
line-follower.py

This codes send commands for tracing black lines in GUIDED_NOGPS mode using DroneKit Python.

"""

from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil # Needed for command message definitions
import time
import math

from line_detector import LineDetector

#Set up option parsing to get connection string
import argparse  
parser = argparse.ArgumentParser(description='Control Copter and send commands in GUIDED mode ')
parser.add_argument('--connect', 
                   help="Vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect
sitl = None


#Start SITL if no connection string specified
if not connection_string:
    import dronekit_sitl
    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()


# Connect to the Vehicle
print 'Connecting to vehicle on: %s' % connection_string
vehicle = connect(connection_string, wait_ready=True)


def arm_and_takeoff_nogps(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude without GPS data.
    """
    
    ##### CONSTANTS #####
    DEFAULT_TAKEOFF_THRUST = 0.65
    SMOOTH_TAKEOFF_THRUST = 0.55
    
    print "Basic pre-arm checks"
    # Don't let the user try to arm until autopilot is ready
    # If you need to disable the arming check, just comment it with your own responsibility.
    #while not vehicle.is_armable:
    #    print " Waiting for vehicle to initialise..."
    #    time.sleep(1)
    
    
    print "Arming motors"
    # Copter should arm in GUIDED_NOGPS mode
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    vehicle.armed = True
    
    while not vehicle.armed:
        print " Waiting for arming..."
        time.sleep(1)
    
    print "Taking off!"
    # Wait until the vehicle reaches a safe height before processing the others
    thrust = DEFAULT_TAKEOFF_THRUST
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        print " Altitude: ", current_altitude
        if current_altitude >= aTargetAltitude*0.95: # Trigger just below target alt.
            print "Reached target altitude"
            break
        elif current_altitude >= aTargetAltitude*0.6:
            thrust = SMOOTH_TAKEOFF_THRUST
        set_attitude(thrust = thrust)
        time.sleep(0.1)
    set_attitude(thrust = 0.5)


def set_attitude(roll_angle = 0.0, pitch_angle = 0.0, yaw_rate = 0.0, thrust = 0.5, duration = 0):
    """
    Note that from AC3.3 the message should be re-sent every second (after about 3 seconds
    with no message the velocity will drop back to zero). In AC3.2.1 and earlier the specified
    velocity persists until it is canceled. The code below should work on either version
    (sending the message multiple times does not cause problems).
    """
    
    """
    The roll and pitch rate cannot be controllbed with rate in radian in AC3.4.4 or earlier,
    so you must use quaternion to control the pitch and roll for those vehicles.
    """
    
    # Thrust >  0.5: Ascend
    # Thrust == 0.5: Hold the altitude
    # Thrust <  0.5: Descend
    msg = vehicle.message_factory.set_attitude_target_encode(
                                                             0,
                                                             0,                                         #target system
                                                             0,                                         #target component
                                                             0b00000000,                                #type mask: bit 1 is LSB
                                                             to_quaternion(roll_angle, pitch_angle),    #q
                                                             0,                                         #body roll rate in radian
                                                             0,                                         #body pitch rate in radian
                                                             math.radians(yaw_rate),                    #body yaw rate in radian
                                                             thrust)                                    #thrust
    vehicle.send_mavlink(msg)
                                                             
    if duration != 0:
        # Divide the duration into the frational and integer parts
        modf = math.modf(duration)
        
        # Sleep for the fractional part
        time.sleep(modf[0])
        
        # Send command to vehicle on 1 Hz cycle
        for x in range(0,int(modf[1])):
            time.sleep(1)
            vehicle.send_mavlink(msg)
            

def to_quaternion(roll = 0.0, pitch = 0.0, yaw = 0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))
    
    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

"""
Find the line to follow and change the attributes of the vehicle appropriately by setting attitude targets.
"""

# Pitch: Negative = Forward, Positive = Backward
# Roll: Negative = Left, Positive = Right
FORWARD_ANGLE = -3
ROLL_ANGLE = 3

PRECISION = 6
TURNING_ANGLE_RANGE = 60

ld = LineDetector(PRECISION)

arm_and_takeoff_nogps(0.7)

print("!!!!!!!!!!! MOVING START !!!!!!!!!")

dir = ld.getTurnDir()

# Initial Setting
while(dir == PRECISION):
    print("Finding Line")
    set_attitude(yaw_rate = 5, duration = 1)
    dir = ld.getTurnDir()

while True:
    # getTurnDir returns
    dir = ld.getTurnDir()
    print("getTurnDir returns %d" % dir)
    
    # If there is no line, land the vehicle
    # TODO: Count some number of PRECISION for correction of the line detection to prevent unexpected landing
    if dir == PRECISION:
        break

    set_attitude(pitch_angle = FORWARD_ANGLE, roll_angle = dir, yaw_rate = TURNING_ANGLE_RANGE/PRECISION*dir, duration = 1)

print("!!!!!!!!!!! MOVING END !!!!!!!!!")

# Set the vehicle to hold the position
#set_attitude(thrust = 0, duration = 3)


"""
The example is completing. LAND at current location.
"""

time.sleep(1)
print("Setting LAND mode...")
vehicle.mode = VehicleMode("LAND")
time.sleep(1)

#Close vehicle object before exiting script
print "Close vehicle object"
vehicle.close()

# Shut down simulator if it was started.
if sitl is not None:
    sitl.stop()

print("Completed")
