.PHONY: test
test:
	pytest -q

print-test-spec:
	python scripts/describe_tests.py tests/test_core/test_interface.py
