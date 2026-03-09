.PHONY: dev dev-backend dev-frontend test build up down clean \
        sim-p2-constrained sim-p2-unconstrained

dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

dev:
	$(MAKE) -j2 dev-backend dev-frontend

test:
	cd backend && python -m pytest tests/ -v

sim-p2-constrained:
	cd backend && python run_p2_all_constrained.py

sim-p2-unconstrained:
	cd backend && python run_p2_all_unconstrained.py

# Future extension (deferred): sim-p2-two-pop — mixed constrained/unconstrained population

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

clean:
	rm -rf data/*.db data/reports data/weights
	rm -rf frontend/node_modules frontend/dist
	rm -rf backend/__pycache__
