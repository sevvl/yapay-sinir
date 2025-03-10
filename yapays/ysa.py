import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Giriş verileri ve hedef çıktılar
x = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

y = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Kullanıcıdan gizli katmandaki nöron sayısını alalım
ara_hucre = int(input("Gizli katmandaki nöron sayısını giriniz: "))


# Sigmoid fonksiyonu
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Ağırlık çarpı çıkış ile net değeri ve aktivasyonlu çıkış değeri hesaplama fonksiyonu
def hesapla_net_ve_aktivasyon(girisler, agirliklar):
    net_deger = np.dot(girisler, agirliklar)
    aktivasyon_deger = sigmoid(net_deger)
    return net_deger, aktivasyon_deger


# Ağırlıkları random üretme fonksiyonu (-3 ile 5 arasında, 0 hariç)
def random_agirliklar(boyut):
    agirliklar = np.random.uniform(-3, 5, boyut)
    # 0 değerlerini tekrar üretmek için
    while np.any(agirliklar == 0):
        agirliklar[agirliklar == 0] = np.random.uniform(-3, 5, np.sum(agirliklar == 0))
    return agirliklar


# Giriş katmanından gizli katmana ağırlıklar (4 giriş, kullanıcı tarafından belirlenen gizli nöron sayısı)
ag_giris_gizli = random_agirliklar((x.shape[1], ara_hucre))

# Gizli katmandan çıkış katmanına ağırlıklar (kullanıcı tarafından belirlenen gizli nöron sayısı, 2 çıkış nöronu)
ag_gizli_cikis = random_agirliklar((ara_hucre, y.shape[1]))

# Tüm girişler için döngü ile net ve aktivasyon hesaplama
for i in range(x.shape[0]):
    print(f"\n--- {i + 1}. Ağ (giriş) için hesaplamalar ---")

    # 1. Giriş katmanından gizli katmana net ve aktivasyon hesaplama
    net_gizli, aktivasyon_gizli = hesapla_net_ve_aktivasyon(x[i], ag_giris_gizli)
    print("Gizli katman net değerleri:", net_gizli)
    print("Gizli katman aktivasyon değerleri:", aktivasyon_gizli)

    # 2. Gizli katmandan çıkış katmanına net ve aktivasyon hesaplama
    net_cikis, aktivasyon_cikis = hesapla_net_ve_aktivasyon(aktivasyon_gizli, ag_gizli_cikis)
    print("Çıkış katmanı net değerleri:", net_cikis)
    print("Çıkış katmanı aktivasyon değerleri:", aktivasyon_cikis)

    # O1 için hata hesaplama
    o1_tahmin = aktivasyon_cikis[0]  # O1 çıkışı
    o1_beklenen = y[i][0]  # O1 için beklenen çıkış
    o1_hata = o1_tahmin * (1 - o1_tahmin) * (o1_beklenen - o1_tahmin)
    print(f"O1 için hata değeri: {o1_hata}")

    # O2 için hata hesaplama
    o2_tahmin = aktivasyon_cikis[1]  # O2 çıkışı
    o2_beklenen = y[i][1]  # O2 için beklenen çıkış
    o2_hata = o2_tahmin * (1 - o2_tahmin) * (o2_beklenen - o2_tahmin)
    print(f"O2 için hata değeri: {o2_hata}")

    # H1 ve H2 için hata hesaplama
    h1_hata = aktivasyon_gizli[0] * (1 - aktivasyon_gizli[0]) * (ag_gizli_cikis[0, :] @ np.array([o1_hata, o2_hata]))
    h2_hata = aktivasyon_gizli[1] * (1 - aktivasyon_gizli[1]) * (ag_gizli_cikis[1, :] @ np.array([o1_hata, o2_hata]))

    print(f"H1 için hata değeri: {h1_hata}")
    print(f"H2 için hata değeri: {h2_hata}")

# Üretilen rastgele ağırlıkları yazdırmak isterseniz:
print("\nRastgele üretilen giriş-gizli katman ağırlıkları:\n", ag_giris_gizli)
print("Rastgele üretilen gizli-çıkış katman ağırlıkları:\n", ag_gizli_cikis)


def w_degerleri(m, n):
    return np.random.uniform(0, 5, size=(m, n))  # m x n boyutunda rastgele ağırlık matris


# Görselleştirme için bir sinir ağı oluşturma fonksiyonu
def ysa_ciz(x_row, y_row, ara_hucre):
    G = nx.DiGraph()  # Yönlü bir grafik

    # Giriş katmanı
    for i, val in enumerate(x_row):
        G.add_node(f'X{i + 1}: {val}', pos=(i, 0))  # Giriş nöronları yukarıdan aşağıya sıralandı

    # Gizli katman
    for i in range(ara_hucre):
        G.add_node(f'H{i + 1}', pos=(i, 1))  # Gizli katman nöronları yukarıdan aşağıya sıralandı

    # Çıkış katmanı
    for i, val in enumerate(y_row):
        G.add_node(f'Y{i + 1}: {val}', pos=(i, 2))  # Çıkış nöronları yukarıdan aşağıya sıralandı

    # Ağırlık matrisini oluştur
    agirliklar = w_degerleri(len(x_row), ara_hucre)  # Giriş katmanından gizli katmana
    agirliklar_cik = w_degerleri(ara_hucre, len(y_row))  # Gizli katmandan çıkış katmanına

    # Giriş katmanından gizli katmana ok
    for i in range(len(x_row)):
        for j in range(ara_hucre):
            G.add_edge(f'X{i + 1}: {x_row[i]}', f'H{j + 1}', weight=agirliklar[i, j])

    # Gizli katmandan çıkış katmanına ok
    for i in range(ara_hucre):
        for j in range(len(y_row)):  # Çıkış hücrelerinin sayısı y_row'un uzunluğuna göre ayarlandı
            G.add_edge(f'H{i + 1}', f'Y{j + 1}: {y_row[j]}', weight=agirliklar_cik[i, j])

    return G


# Ağları görselleştirme
fig, axes = plt.subplots(1, 4, figsize=(20, 10))  # Her biri için 4 ağ olacak

for idx, (x_row, y_row) in enumerate(zip(x, y)):  # x ve y satırlarını birlikte kullanıyoruz
    G = ysa_ciz(x_row, y_row, ara_hucre)
    pos = nx.get_node_attributes(G, 'pos')

    plt.sca(axes[idx])
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrows=True)

    # Ağırlıkları ekrana yazdır
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # plt.title(f'Yapay Sinir Ağı {idx+1}')

plt.tight_layout()
plt.show()