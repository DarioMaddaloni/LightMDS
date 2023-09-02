Abbiamo un'immagine con #pixels pixels.
Abbiamo due generatori di punti:
	un cerchio di distribuizione normale con parametri mu=0 e sigma=s (che dovremo di volta in volta stimare -> N(0, s)
	l'immagine stessa che supponiamo essere una distribuzione normale di cui non conosciamo il parametro p -> U(p)

Faccio una stima iniziale del cerchio: cx, cy, r
Calcolo dk=|(xk-cx)**2+(yk-cy)**2-r**2| per ogni punto nell'immagine
Stimo sigma e p
