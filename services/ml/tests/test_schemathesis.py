"""Property-based conformance: generated requests never crash the service and every
response matches the contract. Runs in process, no network."""

from typing import Any

import numpy as np
import schemathesis
from numpy.typing import NDArray
from schemathesis.checks import not_a_server_error
from schemathesis.specs.openapi.checks import (
    content_type_conformance,
    response_schema_conformance,
    status_code_conformance,
)

from agrofarm_ml.leaves.model import LeafModel
from agrofarm_ml.main import create_app


class TestLeafPredictor:
    """The optional image route needs a test-only model for generated valid requests."""

    __test__ = False

    def predict(self, data: NDArray[np.float64]) -> Any:
        return np.array([[0.2] * 5 for _ in data])


schema = schemathesis.openapi.from_asgi(
    "/openapi.json",
    create_app(leaf_model=LeafModel(TestLeafPredictor(), {"model_version": "test-only"})),
)


@schema.parametrize()
def test_service_conforms_to_the_contract(case: "schemathesis.Case[Any]") -> None:
    case.call_and_validate(
        checks=[
            not_a_server_error,
            status_code_conformance,
            content_type_conformance,
            response_schema_conformance,
        ]
    )
