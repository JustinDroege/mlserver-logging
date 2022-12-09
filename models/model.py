from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
import logging


class CustomModel(MLModel):
    
    async def load(self) -> bool:
        
        for name in logging.root.manager.loggerDict:
            for handler in logging.getLogger(name).handlers:
                print(f"Logger: {name}, Handler: {handler.name}")
                print(f"Logger: {name}, Formatter: {handler.formatter}")


        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        outputs = []
        
        for name in logging.root.manager.loggerDict:
            for handler in logging.getLogger(name).handlers:
                print(f"Logger: {name}, Handler: {handler.name}")
                print(f"Logger: {name}, Formatter: {handler.formatter}")

        for request_input in payload.inputs:
            outputs.append(
                    ResponseOutput(
                        name=request_input.name,
                        datatype=request_input.datatype,
                        shape=request_input.shape,
                        data=request_input.data
                    )
                )
        
        return InferenceResponse(model_name=self.name, outputs=outputs)

