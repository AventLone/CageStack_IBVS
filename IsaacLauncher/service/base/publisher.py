from ecal.core.publisher import MessagePublisher


class BytesPublisher(MessagePublisher):
  """Spezialized publisher that sends out plain strings
  """
  def __init__(self, name):
    topic_type = "base:std::string"
    topic_desc = b""
    super(BytesPublisher, self).__init__(name, topic_type, topic_desc)

  def send(self, msg, time=-1):
    return self.c_publisher.send(msg, time)

  def send_sync(self, msg, time, ack_timeout_ms):
    return self.c_publisher.send_sync(msg, time, ack_timeout_ms)