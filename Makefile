.PHONY: up build proddown test clean

up:
	docker-compose -f docker-compose.dev.yml up

build:
	docker-compose -f docker-compose.dev.yml build

prod:
	docker-compose -f docker-compose.yml build

down:
	docker-compose -f docker-compose.dev.yml down

test:
	docker-compose -f docker-compose.test.yml run --rm test pytest tests/ -v

clean:
	docker-compose -f docker-compose.dev.yml down --volumes --remove-orphans
	docker-compose -f docker-compose.test.yml down --volumes --remove-orphans



