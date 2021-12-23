import numpy as np




def data_symptom(demo,all_symptom,all_symptom_name):
    #salut
    age=demo["age"].to_numpy()
    gender = demo["Gender"].to_numpy()
    education = demo["education"].to_numpy()
    illduration = demo["illduration"].to_numpy()
    all_symptom.append(illduration)
    all_symptom_name.append("illduration")
    visualAcuity = demo["VisualAcuity"].to_numpy()
    all_symptom.append(visualAcuity)
    all_symptom_name.append("VisualAcuity")
    #spqi,spqii,spqiii empty. Why ?
    sans = demo["SANS"].to_numpy() #Important
    all_symptom.append(sans)
    all_symptom_name.append("SANS")
    affect= demo["Affect"].to_numpy()
    all_symptom.append(affect)
    all_symptom_name.append("Affect")
    alogie= demo["Alogie"].to_numpy()
    all_symptom.append(alogie)
    all_symptom_name.append("Alogie")
    abulie= demo["Abulie"].to_numpy()
    all_symptom.append(abulie)
    all_symptom_name.append("Abulie")
    anhedonie= demo["Anhedonie"].to_numpy()
    all_symptom.append(anhedonie)
    all_symptom_name.append("Anhedonie")
    attention= demo["Aufmerksamkeit"].to_numpy()
    all_symptom.append(attention)
    all_symptom_name.append("Aufmerksamkeit")
    saps = demo["SAPS"].to_numpy() #Important
    all_symptom.append(saps)
    all_symptom_name.append("SAPS")
    Hallu=demo['Halluzinationen'].to_numpy()
    all_symptom.append(Hallu)
    all_symptom_name.append("Halluzinationen")
    folie= demo["Wahnerleben"].to_numpy()
    all_symptom.append(folie)
    all_symptom_name.append("Wahnerleben")
    bizzares = demo["Bizarres Verhalten"].to_numpy()
    all_symptom.append(bizzares)
    all_symptom_name.append("Bizarres Verhalten")
    pense = demo["Denkstorung"].to_numpy()
    all_symptom.append(pense)
    all_symptom_name.append("Denkstorung")
    cpz = demo["CPZ"].to_numpy()
    all_symptom.append(cpz)
    all_symptom_name.append("CPZ")
    handedness = demo["Handedness"].to_numpy()
    all_symptom.append(handedness)
    all_symptom_name.append("Handedness")
    #Missing a lot of data in the 4 last columns
    return all_symptom,all_symptom_name
