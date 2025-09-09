import argparse
import asyncio
import time

import numpy as np
import torch
import kornia as K
from google.protobuf.descriptor_pb2 import FileDescriptorProto, FileDescriptorSet
from google.protobuf.timestamp_pb2 import Timestamp

from foxglove_websocket.server import FoxgloveServer
from foxglove_websocket import run_cancellable
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig, SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.oak import oak_pb2
from google.protobuf.empty_pb2 import Empty
from kornia_rs import ImageDecoder

def build_file_descriptor_set(message_cls) -> FileDescriptorSet:
    """Package up the protobuf schema (and its imports) so Foxglove knows how to decode it."""
    fds = FileDescriptorSet()

    # Helper to turn a FileDescriptor into a FileDescriptorProto
    def append_fd(fd):
        fdp = FileDescriptorProto()
        fdp.ParseFromString(fd.serialized_pb)
        fds.file.append(fdp)

    # Append the message’s own .proto file...
    append_fd(message_cls.DESCRIPTOR.file)
    # ...and all of its imported dependencies
    for dep in message_cls.DESCRIPTOR.file.dependencies:
        append_fd(dep)

    return fds

def decode_disparity(msg: oak_pb2.OakFrame, decoder: ImageDecoder) -> torch.Tensor:
    dl = decoder.decode(msg.image_data)
    return torch.from_dlpack(dl)[..., 0].float()

def get_camera_matrix(cam: oak_pb2.CameraData) -> torch.Tensor:
    fx, fy = cam.intrinsic_matrix[0], cam.intrinsic_matrix[4]
    cx, cy = cam.intrinsic_matrix[2], cam.intrinsic_matrix[5]
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

async def main():
    # parse your args
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-config", required=True, type=argparse.FileType("r"))
    args = parser.parse_args()

    # 1) start Foxglove WebSocket server
    server = FoxgloveServer(host="0.0.0.0", port=8765, name="amiga-pointcloud-server")
    pc_fds = build_file_descriptor_set(PointCloud)
    pc_channel_id = await server.add_channel({
        "topic": "/oak/pointcloud",
        "schemaName": PointCloud.DESCRIPTOR.full_name,
        "schema": pc_fds.SerializeToString(),
        "schemaEncoding": "proto",
    })

    # 2) set up your Amiga camera client
    config = proto_from_json_file(args.service_config, EventServiceConfig())
    cam_client = EventClient(config)
    calib: oak_pb2.OakCalibration = await cam_client.request_reply("/calibration", Empty(), decode=True)
    cam_data = calib.camera_data[0]
    cam_mat = get_camera_matrix(cam_data)
    decoder = ImageDecoder()
    baseline = 0.075
    focal = float(cam_mat[0, 0])

    # 3) run both the Foxglove server and your camera loop
    async with run_cancellable(server.serve()):
        async for _, frame in cam_client.subscribe(
            SubscribeRequest(uri="oak/1/disparity", every_n=5),
            decode=True,
        ):
            # decode + backproject
            disp = decode_disparity(frame, decoder)
            depth = K.geometry.depth.depth_from_disparity(disp, baseline, focal)
            xyz = K.geometry.depth.depth_to_3d_v2(depth, cam_mat)
            mask = (xyz[..., 2:] >= 0.2) & (xyz[..., 2:] <= 7.5)
            pts = xyz[mask.repeat(1, 1, 3)].reshape(-1, 3).cpu().numpy().astype(np.float32)

            # timestamp in nanoseconds
            ts_ns = time.time_ns()
            # also stamp the embedded protobuf Timestamp field
            ts = Timestamp(seconds=ts_ns // 1_000_000_000, nanos=ts_ns % 1_000_000_000)

            # build and send the PointCloud message
            pc_msg = PointCloud(
                timestamp=ts,
                frame_id="oak_frame",
                point_stride=12,
                fields=[
                    PackedElementField(name="x", offset=0, type=PackedElementField.FLOAT32),
                    PackedElementField(name="y", offset=4, type=PackedElementField.FLOAT32),
                    PackedElementField(name="z", offset=8, type=PackedElementField.FLOAT32),
                ],
                data=pts.tobytes(),
            )
            await server.send_message(pc_channel_id, ts_ns, pc_msg.SerializeToString())

if __name__ == "__main__":
    asyncio.run(main())
