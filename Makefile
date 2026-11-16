# Root targets fan out to the two services. Each service is checked on its own.
.PHONY: dev test lint typecheck build check check-ml check-data audit
SERVICES := ml data

dev: ## both services locally
	docker compose up --build

check-ml:
	$(MAKE) -C services/ml check

check-data:
	$(MAKE) -C services/data check

check: check-ml check-data ## what CI runs when root files change

test lint typecheck build audit:
	@for s in $(SERVICES); do $(MAKE) -C services/$$s $@ || exit 1; done
