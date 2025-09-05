'''
    Functions for handling LLM calling
'''
from typing import Optional
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process

class SGLangServer:
    def __init__(
        self,
        host: str,
        port: int,
        model_path: str,
        served_model_name: str,
        tensor_parallel: Optional[int] = None
    ):
        '''
            Assigning the arguments to the class variables
        '''
        self.host = host
        self.port = port
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.tensor_parallel = tensor_parallel
        self.server_process = None

    def launch_server(self) -> bool:
        '''
            Launch the server
        '''

        # initializing the server cmd
        server_cmd = f''' python -m sglang.launch_server \
            --host {self.host} \
            --port {self.port} \
            --model-path {self.model_path} \
            --served-model-name {self.served_model_name}
        '''
        if self.tensor_parallel is not None:
            server_cmd += f' --tensor-parallel {self.tensor_parallel}'
        self.server_process, _ = launch_server_cmd(server_cmd)

        wait_for_server('http://{}:{}'.format(self.host, self.port))
        return True
    
    def terminate_server(self) -> bool:
        '''
            Terminate the server
        '''
        if self.server_process is not None:
            terminate_process(self.server_process)
            self.server_process = None
        return True
    
if __name__ == '__main__':
    server = SGLangServer(
        host='localhost',
        port=8000,
        model_path='/home/inair/data/models/Llama-3-8B-Instruct',
        served_model_name='Llama-3-8B-Instruct',
        tensor_parallel=2
    )
    server.launch_server()
    import pdb; pdb.set_trace()
    server.terminate_server()