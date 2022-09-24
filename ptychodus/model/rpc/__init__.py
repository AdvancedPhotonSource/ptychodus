from .core import RPCCore, RPCMessageService


def main() -> int:
    import argparse
    from .client import RPCMessageClient

    parser = argparse.ArgumentParser(
        prog='ptychodus-rpc',
        description='ptychodus-rpc communicates with an active ptychodus process')
    parser.add_argument('-m', '--message', action='store', required=True, \
            help='message to send')
    parser.add_argument('-p', '--port', action='store', type=int, default=9999, \
            help='remote process communication port number')
    args = parser.parse_args()

    client = RPCMessageClient(args.port)
    response = client.send(args.message)

    print(response)

    return 0
