# Fonction de récompense — Microban Velocity

La récompense totale à chaque pas de simulation est la **somme pondérée** de tous les termes :

$$r = \sum_i w_i \cdot r_i$$

Chaque terme peut être une **récompense positive** (encourage un comportement) ou une **pénalité** (poids négatif, décourage un comportement).

---

## Termes de suivi de commande

### `track_linear_velocity` — poids `+2.0`

Récompense le robot pour suivre la vitesse linéaire commandée (plan XY).

$$r = \exp\!\left(-\frac{\|\mathbf{v}_{xy}^{cmd} - \mathbf{v}_{xy}\|^2 + v_z^2}{\sigma^2}\right)$$

- $\mathbf{v}_{xy}^{cmd}$ : vitesse linéaire XY commandée  
- $\mathbf{v}_{xy}$ : vitesse linéaire XY mesurée dans le repère corps  
- $v_z$ : vitesse verticale (supposée nulle → pénalisée si non nulle)  
- $\sigma = \sqrt{0.1}$ (Microban)

Le noyau gaussien donne `1.0` si la vitesse est parfaitement suivie, et décroît vers `0` avec l'erreur.

---

### `track_angular_velocity` — poids `+2.0`

Récompense le suivi de la vitesse angulaire en lacet (yaw).

$$r = \exp\!\left(-\frac{(\omega_z^{cmd} - \omega_z)^2 + \|\boldsymbol{\omega}_{xy}\|^2}{\sigma^2}\right)$$

- $\omega_z^{cmd}$ : vitesse angulaire en lacet commandée  
- $\omega_z$ : vitesse angulaire en lacet mesurée  
- $\boldsymbol{\omega}_{xy}$ : rotations parasites en roulis/tangage (supposées nulles)  
- $\sigma = \sqrt{0.5}$

---

## Termes de posture

### `upright` — poids `+1.0`

Récompense l'orientation verticale du torse (robot droit).

$$r = \exp\!\left(-\frac{\|g_{xy}^{proj}\|^2}{\sigma^2}\right)$$

- $\mathbf{g}^{proj}$ : vecteur gravité projeté dans le repère du torse  
- Les composantes XY de ce vecteur sont nulles si le torse est parfaitement vertical  
- $\sigma = \sqrt{0.2}$

---

### `pose` — poids `+1.0`

Récompense la proximité de chaque articulation par rapport à sa position nominale, avec une tolérance qui dépend de la vitesse de déplacement.

$$r = \exp\!\left(-\frac{1}{N}\sum_i \frac{(q_i - q_i^{default})^2}{\sigma_i^2}\right)$$

Trois régimes de vitesse (`total\_speed = \|v_{xy}^{cmd}\| + |\omega_z^{cmd}|`) :

| Régime | Condition | Tolérance (std par défaut) |
|--------|-----------|---------------------------|
| **Debout** | speed < 0.01 | Faible (contrainte stricte) |
| **Marche** | 0.01 ≤ speed < 1.5 | Modérée |
| **Course** | speed ≥ 1.5 | Large (mouvements amples autorisés) |

Les std sont définis par pattern de joint (voir `microban_velocity_env_cfg.py`). Exemples :
- `hip_pitch`, `knee` : std_walking = 0.4 (beaucoup de mouvement autorisé)
- `ankle_roll`, `hip_yaw` : std_walking = 0.2 (moins de mouvement nécessaire)
- `shoulder_pitch`, `elbow` : std_standing = 0.1 (bras proche de la position neutre à l'arrêt)

---

## Termes de régularisation du corps

### `body_ang_vel` — poids `-0.05`

Pénalise les vitesses angulaires excessives du torse en roulis et tangage.

$$\text{coût} = \omega_{roll}^2 + \omega_{pitch}^2$$

Décourage les oscillations du torse pendant la marche.

---

### `angular_momentum` — poids `-0.02`

Pénalise la magnitude du moment cinétique total du robot.

$$\text{coût} = \|\mathbf{L}\|^2$$

- $\mathbf{L}$ : moment cinétique global du robot (calculé par MuJoCo)  
- Décourage les mouvements de bras anarchiques et encourage un balancement naturel.

---

## Termes sur les limites articulaires

### `dof_pos_limits` — poids `-1.0`

Pénalise les dépassements des limites articulaires "douces" (soft limits, légèrement en deçà des limites mécaniques).

$$\text{coût} = \sum_i \max(0,\, q_i - q_i^{max,soft}) + \max(0,\, q_i^{min,soft} - q_i)$$

---

## Termes de régularisation des actions

### `action_rate_l2` — poids `-0.5`

Pénalise les changements brusques d'actions entre deux pas de temps consécutifs.

$$\text{coût} = \sum_i (a_i^t - a_i^{t-1})^2$$

- $a^t$ : sortie brute de la politique au pas courant  
- Encourage des commandes articulaires lisses.

---

## Termes de qualité de démarche

### `air_time` — poids `+0.1`

Récompense les pieds qui restent en l'air pendant une durée dans la plage `[threshold\_min, threshold\_max]`.

$$r = \sum_{foot} \mathbf{1}\left[t_{air} \in [0.10\text{ s},\; 0.25\text{ s}]\right]$$

- Actif seulement si la commande de vitesse dépasse `command_threshold = 0.01`  
- Encourage un cycle de marche avec des pas bien distincts (ni trop courts ni trop longs).

---

### `foot_clearance` — poids `-2.0`

Pénalise les pieds qui ne passent pas à la hauteur cible pendant leur phase de vol, proportionnellement à leur vitesse horizontale.

$$\text{coût} = \sum_{foot} |z_{foot} - h_{target}| \cdot \|\mathbf{v}_{xy,foot}\|$$

- $h_{target} = 0.02\text{ m}$ (Microban)  
- Actif seulement si la commande > `command_threshold = 0.01`  
- Un pied qui se déplace vite mais à la mauvaise hauteur est fortement pénalisé.

---

### `foot_swing_height` — poids `-0.25`

Pénalise le pic de hauteur d'un pied lors de son envol si celui-ci diffère de la hauteur cible. La pénalité est appliquée **au moment de l'atterrissage**.

$$\text{coût} = \sum_{foot} \left(\frac{h_{peak}}{h_{target}} - 1\right)^2 \cdot \mathbf{1}_{first\_contact}$$

- $h_{target} = 0.02\text{ m}$ (Microban)  
- Actif seulement si la commande > `command_threshold = 0.01`  
- Complément à `foot_clearance` : ici on évalue la hauteur maximale atteinte, pas instantanée.

---

### `foot_slip` — poids `-0.1`

Pénalise le glissement des pieds sur le sol (vitesse XY pendant le contact).

$$\text{coût} = \sum_{foot} \|\mathbf{v}_{xy,foot}\|^2 \cdot \mathbf{1}_{in\_contact}$$

- Actif seulement si la commande > `command_threshold = 0.01`  
- Encourage des appuis stables et un transfert de poids propre.

---

### `soft_landing` — poids `-1e-5`

Pénalise les forces d'impact élevées à l'atterrissage des pieds.

$$\text{coût} = \sum_{foot} \|\mathbf{F}\| \cdot \mathbf{1}_{first\_contact}$$

- Poids très faible : signal de régularisation doux  
- Encourage des atterrissages "en douceur".

---

## Résumé des poids (Microban)

| Terme | Type | Poids |
|-------|------|-------|
| `track_linear_velocity` | Récompense | +2.0 |
| `track_angular_velocity` | Récompense | +2.0 |
| `upright` | Récompense | +1.0 |
| `pose` | Récompense | +1.0 |
| `air_time` | Récompense | +0.1 |
| `foot_clearance` | Pénalité | -2.0 |
| `dof_pos_limits` | Pénalité | -1.0 |
| `action_rate_l2` | Pénalité | -0.5 |
| `foot_swing_height` | Pénalité | -0.25 |
| `body_ang_vel` | Pénalité | -0.05 |
| `foot_slip` | Pénalité | -0.1 |
| `angular_momentum` | Pénalité | -0.02 |
| `soft_landing` | Pénalité | -1e-5 |
