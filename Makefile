.PHONY: test
test:
	pytest -q

print-test-spec:
	python scripts/describe_tests.py tests/test_core/test_interface.py

print-test-spec-core:
	python scripts/describe_tests.py tests/test_core/test_interface.py
	python scripts/describe_tests.py tests/test_core/test_config.py
	python scripts/describe_tests.py tests/test_core/test_registry.py
