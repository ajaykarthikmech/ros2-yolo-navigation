import rclpy
from rclpy.node import Node

class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.timer = self.create_timer(
            1.0,
            self.timer_callback
        )

        self.get_logger().info(
            "Camera Node Started"
        )

    def timer_callback(self):
        self.get_logger().info(
            "Camera Running..."
        )

def main(args=None):

    rclpy.init(args=args)

    node = CameraNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()