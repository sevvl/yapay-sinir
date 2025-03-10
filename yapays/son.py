import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# Sigmoid fonksiyonu
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Türevli sigmoid fonksiyonu (geri yayılımda kullanılacak)
def sigmoid_derivative(z):
    return z * (1 - z)


# Ağırlık çarpı çıkış ile net değeri ve aktivasyonlu çıkış değeri hesaplama fonksiyonu
def hesapla_net_ve_aktivasyon(girisler, agirliklar):
    net_deger = np.dot(girisler, agirliklar)
    aktivasyon_deger = sigmoid(net_deger)
    return net_deger, aktivasyon_deger


# Ağırlıkları random üretme fonksiyonu (-3 ile 5 arasında, 0 hariç)
def random_agirliklar(boyut):
    agirliklar = np.random.uniform(-3, 5, boyut)
    while np.any(agirliklar == 0):
        agirliklar[agirliklar == 0] = np.random.uniform(-3, 5, np.sum(agirliklar == 0))
    return agirliklar


# Toplam hatayı hesaplama fonksiyonu
def toplam_hata(hedef, cikis):
    return 0.5 * np.sum((hedef - cikis) ** 2)


# Ağ yapısını görselleştirme fonksiyonu
def plot_network(giris_sayisi, gizli_sayisi, cikis_sayisi):
    G = nx.DiGraph()

    # Giriş katmanı düğümlerini ekleyelim
    for i in range(giris_sayisi):
        G.add_node(f"X{i + 1}", pos=(0, i))

    # Gizli katman düğümleri
    for i in range(gizli_sayisi):
        G.add_node(f"H{i + 1}", pos=(1, i))

    # Çıkış katmanı düğümleri
    for i in range(cikis_sayisi):
        G.add_node(f"O{i + 1}", pos=(2, i))

    # Giriş katmanından gizli katmana bağlantılar
    for i in range(giris_sayisi):
        for j in range(gizli_sayisi):
            G.add_edge(f"X{i + 1}", f"H{j + 1}")

    # Gizli katmandan çıkış katmanına bağlantılar
    for i in range(gizli_sayisi):
        for j in range(cikis_sayisi):
            G.add_edge(f"H{i + 1}", f"O{j + 1}")

    # Düğüm pozisyonları
    pos = nx.get_node_attributes(G, 'pos')

    # Ağ çizimi
    plt.figure(figsize=(10, 5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold',
            arrows=True)

    # Kenarların üzerine ağırlıklar ekleyebiliriz (örnek olarak rastgele değerler ekleyebiliriz):
    labels = {(u, v): f'{round(np.random.uniform(-1, 1), 2)}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')

    plt.title("Yapay Sinir Ağı Yapısı")
    plt.show()


# Eğitim verileri ve hedef çıktılar
x = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

y = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Parametreler
ogrenme_katsayisi = 0.1
iterasyon_sayisi = 10000

# Ağırlıkları başlat
ag_giris_gizli = random_agirliklar((x.shape[1], 4))  # Girişten gizli katmana (4 nöron)
ag_gizli_cikis = random_agirliklar((4, y.shape[1]))  # Gizliden çıkış katmanına

# Eğitim döngüsü
for epoch in range(iterasyon_sayisi):
    toplam_hata_degeri = 0

    for i in range(x.shape[0]):
        # İleri besleme
        net_gizli, aktivasyon_gizli = hesapla_net_ve_aktivasyon(x[i], ag_giris_gizli)
        net_cikis, aktivasyon_cikis = hesapla_net_ve_aktivasyon(aktivasyon_gizli, ag_gizli_cikis)

        # Hata hesaplama
        hata = y[i] - aktivasyon_cikis
        toplam_hata_degeri += toplam_hata(y[i], aktivasyon_cikis)

        # Geri yayılım
        delta_cikis = hata * sigmoid_derivative(aktivasyon_cikis)
        delta_gizli = np.dot(delta_cikis, ag_gizli_cikis.T) * sigmoid_derivative(aktivasyon_gizli)

        # Ağırlık güncelleme
        ag_gizli_cikis += ogrenme_katsayisi * np.dot(aktivasyon_gizli[:, None], delta_cikis[None, :])
        ag_giris_gizli += ogrenme_katsayisi * np.dot(x[i][:, None], delta_gizli[None, :])

    # Her 1000 iterasyonda toplam hatayı ekrana yazdır
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Toplam Hata: {toplam_hata_degeri}")

# Eğitim tamamlandıktan sonra toplam hatayı yazdır
print("\nEğitim tamamlandı. Toplam Hata:", toplam_hata_degeri)

# Ağı görselleştir
plot_network(giris_sayisi=4, gizli_sayisi=4, cikis_sayisi=2)
