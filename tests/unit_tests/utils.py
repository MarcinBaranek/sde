import pytest


@pytest.fixture(params=['float16', 'float32', 'float64'])
def precision(request):
    return request.param


tolerance = {
    'float16': 1.e-2,
    'float32': 1.e-4,
    'float64': 1.e-8,
    'float128': 1.e-12,
}
