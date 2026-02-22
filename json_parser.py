import json
import struct


def recup_poids_biais(path):
    try:
        with open(path, "r") as file:
            global data
            data = json.load(file)

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file.")

    nombre_transitions_lineaires = len(data["item"]) - 1
    matrices_poids = [[[]] for i in range(2 * nombre_transitions_lineaires)]
    for i in range(nombre_transitions_lineaires):
        octets = data["item"]["linear" + str(i + 1)]["weight"]["param"]["bytes"]
        [colonnes, lignes] = data["item"]["linear" + str(i + 1)]["weight"]["param"][
            "shape"
        ]
        matrice_transition = [[0 for i in range(colonnes)] for j in range(lignes)]
        poids = [0 for i in range(0, len(octets) // 4)]
        for j in range(len(octets) // 4):
            poids[j] = struct.unpack("<f", bytes(octets[4 * j : 4 * j + 4]))[0]
        for j in range(lignes):
            matrice_transition[j] = poids[colonnes * j : colonnes * (j + 1)]
        matrices_poids[2 * i] = matrice_transition
    matrices_poids = matrices_poids[:-1]

    vecteurs_biais = [[] for i in range(2 * nombre_transitions_lineaires)]
    for i in range(nombre_transitions_lineaires):
        octets = data["item"]["linear" + str(i + 1)]["bias"]["param"]["bytes"]
        taille = data["item"]["linear" + str(i + 1)]["bias"]["param"]["shape"][0]
        vecteur = [0 for i in range(taille)]
        for j in range(len(octets) // 4):
            vecteur[j] = struct.unpack("<f", bytes(octets[4 * j : 4 * j + 4]))[0]
        vecteurs_biais[2 * i] = vecteur
    vecteurs_biais = vecteurs_biais[:-1]

    return matrices_poids, vecteurs_biais
