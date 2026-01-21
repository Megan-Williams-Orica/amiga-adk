import asyncio


async def move_backwards(twist, client):
    # Start with the robot not moving for 1 second
    print("The Amiga will remain stationary for 1 second")
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(1.0)

    # Move the robot backwards at 0.5 m/s for 2 seconds
    print("The Amiga will now move backwards for 2 seconds")
    twist.linear_velocity_x = -0.5
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(2.0)

    # Stop the robot indefinitely
    print("The Amiga will now stop indefinitely")
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)


async def move_forwards(twist, client):
    # Start with the robot not moving for 1 second
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(1.0)

    # Move the robot forwards at 0.5 m/s for 2 seconds
    twist.linear_velocity_x = 0.5
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(2.0)

    # Stop the robot indefinitely
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)


async def turn_left(twist, client):
    # Start with the robot not moving for 1 second
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(1.0)

    # Turn the robot left at 0.5 m/s for 2 seconds
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = -0.5
    await client.request_reply("/twist", twist)
    await asyncio.sleep(2.0)

    # Stop the robot indefinitely
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)


async def turn_right(twist, client):
    # Start with the robot not moving for 1 second
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(1.0)

    # Turn the robot right at 0.5 m/s for 2 seconds
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.5
    await client.request_reply("/twist", twist)
    await asyncio.sleep(2.0)

    # Stop the robot indefinitely
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)


async def stop(twist, client):
    # Stop the robot indefinitely
    twist.linear_velocity_x = 0.0
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0
    await client.request_reply("/twist", twist)
    await asyncio.sleep(0.05)